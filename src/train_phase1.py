"""
Phase 1 JEPA Training Pipeline:
Implements Self-Supervised Latent Prediction on State-derived Context-Target Views.
Includes pilot verification, representation collapse monitoring, and model persistence.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import yaml
import torch
import torch.nn as nn
import sys

project_root_dir = Path(__file__).resolve().parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.data_processing import load_and_clean_states, split_states
from src.view_generator import ViewGenerator, ContextTargetPair
from src.jepa_model import TextJEPA
from src.losses import JEPALoss
from src.metrics import MetricsTracker
from src.checkpointing import save_checkpoint, save_reusable_encoder


class JEPADataset(Dataset):
    """PyTorch Dataset yielding context and target text views."""

    def __init__(self, pairs: List[ContextTargetPair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        p = self.pairs[idx]
        return {
            "context_text": p.context_text,
            "target_text": p.target_text,
            "domain": p.domain,
        }


class JEPACollator:
    """Dynamic tokenization collator for context and target views."""

    def __init__(self, tokenizer: AutoTokenizer, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        contexts = [item["context_text"] for item in batch]
        targets = [item["target_text"] for item in batch]

        ctx_encoded = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        tgt_encoded = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        return {
            "context_input_ids": ctx_encoded["input_ids"],
            "context_attention_mask": ctx_encoded["attention_mask"],
            "target_input_ids": tgt_encoded["input_ids"],
            "target_attention_mask": tgt_encoded["attention_mask"],
        }


def run_validation(
    model: TextJEPA,
    val_loader: DataLoader,
    loss_fn: JEPALoss,
    device: torch.device,
    use_autocast: bool,
    dtype: torch.dtype,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Runs evaluation over the validation set without updating parameters or EMA."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    latent_norm_acc = 0.0
    latent_std_acc = 0.0
    pairwise_sim_acc = 0.0
    batch_count = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break

            ctx_ids = batch["context_input_ids"].to(device, non_blocking=True)
            ctx_mask = batch["context_attention_mask"].to(device, non_blocking=True)
            tgt_ids = batch["target_input_ids"].to(device, non_blocking=True)
            tgt_mask = batch["target_attention_mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                z_pred, z_target, z_context = model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
                loss = loss_fn(z_pred, z_target)

            bs = ctx_ids.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            diag = loss_fn.compute_diagnostics(z_context)
            latent_norm_acc += diag.mean_latent_norm
            latent_std_acc += diag.mean_dimension_std
            pairwise_sim_acc += diag.mean_pairwise_similarity
            batch_count += 1

    avg_loss = total_loss / max(total_samples, 1)
    return {
        "val_loss": avg_loss,
        "val_mean_latent_norm": latent_norm_acc / max(batch_count, 1),
        "val_mean_latent_std": latent_std_acc / max(batch_count, 1),
        "val_mean_pairwise_similarity": pairwise_sim_acc / max(batch_count, 1),
    }


def run_pilot(
    cfg: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    """
    Executes a comprehensive pilot training session on a small subset:
    1. Tests forward and backward pass
    2. Verifies finite loss
    3. Confirms context encoder and predictor receive non-zero gradients
    4. Confirms target encoder receives zero ordinary gradients
    5. Confirms EMA update alters target encoder parameters
    6. Saves pilot checkpoint
    7. Measures memory and throughput
    """
    print("=" * 60)
    print("           STARTING PHASE 1 PILOT RUN           ")
    print("=" * 60)

    device_str = "cuda" if torch.cuda.is_available() and cfg["experiment"].get("device", "cuda") == "cuda" else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    pilot_cfg = cfg["pilot"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]

    train_parquet = project_root / data_cfg["train_parquet"]
    records = load_and_clean_states(train_parquet, min_word_count=data_cfg.get("min_words_total", 8))
    print(f"Loaded {len(records)} valid states from train parquet.")

    # Select pilot subset
    pilot_records = records[: pilot_cfg["num_samples"]]
    train_recs, val_recs, _ = split_states(
        pilot_records,
        val_ratio=0.20,
        seed=cfg["experiment"]["seed"],
        deduplicate_identical_states=False,
    )

    view_gen = ViewGenerator(
        strategy=data_cfg.get("strategy", "span_mask"),
        mask_ratio=data_cfg.get("mask_ratio", 0.25),
        min_words_total=data_cfg.get("min_words_total", 8),
        context_ratio=data_cfg.get("context_ratio", 0.60),
        seed=cfg["experiment"]["seed"],
    )
    train_pairs, train_stats = view_gen.generate_pairs_batch(train_recs)
    val_pairs, val_stats = view_gen.generate_pairs_batch(val_recs)

    print(f"Pilot Pairs: {len(train_pairs)} train, {len(val_pairs)} val.")
    assert len(train_pairs) > 0, "No valid training pairs generated in pilot!"

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone_name"])
    collator = JEPACollator(tokenizer, max_seq_length=pilot_cfg["max_seq_length"])

    train_loader = DataLoader(
        JEPADataset(train_pairs),
        batch_size=pilot_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        JEPADataset(val_pairs),
        batch_size=pilot_cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    model = TextJEPA(
        backbone_name_or_path=model_cfg["backbone_name"],
        latent_dim=model_cfg["latent_dim"],
        predictor_hidden_dim=model_cfg["predictor_hidden_dim"],
        predictor_num_layers=model_cfg["predictor_num_layers"],
        pooling_type=model_cfg["pooling_type"],
        normalize_latents=model_cfg["normalize_latents"],
        ema_momentum=model_cfg["ema_momentum"],
        dropout_prob=model_cfg.get("dropout", 0.1),
    ).to(device)

    loss_fn = JEPALoss(
        loss_type=loss_cfg.get("loss_type", "normalized_mse"),
        variance_threshold=loss_cfg.get("variance_threshold", 0.01),
        similarity_threshold=loss_cfg.get("similarity_threshold", 0.95),
        min_mean_std_threshold=loss_cfg.get("min_mean_std_threshold", 0.05),
    )

    param_groups = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW([
        {"params": param_groups["encoder"], "lr": cfg["training"]["encoder_lr"]},
        {"params": param_groups["predictor"], "lr": cfg["training"]["predictor_lr"]},
    ], weight_decay=cfg["training"]["weight_decay"])

    # 1. Verification of target encoder requires_grad=False
    for name, p in model.target_encoder.named_parameters():
        assert not p.requires_grad, f"Target encoder parameter {name} has requires_grad=True!"

    # Record initial target parameter to verify EMA update
    sample_tgt_param = next(model.target_encoder.parameters()).clone()

    model.train()
    step_losses = []
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        if step >= pilot_cfg["num_steps"]:
            break

        ctx_ids = batch["context_input_ids"].to(device)
        ctx_mask = batch["context_attention_mask"].to(device)
        tgt_ids = batch["target_input_ids"].to(device)
        tgt_mask = batch["target_attention_mask"].to(device)

        optimizer.zero_grad()
        z_pred, z_target, z_context = model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
        loss = loss_fn(z_pred, z_target)

        assert torch.isfinite(loss), f"Pilot loss is not finite: {loss.item()}"
        loss.backward()

        # Check gradients
        ctx_grad_norms = [p.grad.norm().item() for p in model.context_encoder.parameters() if p.grad is not None]
        pred_grad_norms = [p.grad.norm().item() for p in model.predictor.parameters() if p.grad is not None]
        tgt_grads = [p.grad for p in model.target_encoder.parameters() if p.grad is not None]

        assert len(ctx_grad_norms) > 0 and sum(ctx_grad_norms) > 0, "Context encoder received zero gradients!"
        assert len(pred_grad_norms) > 0 and sum(pred_grad_norms) > 0, "Predictor received zero gradients!"
        assert len(tgt_grads) == 0, f"Target encoder received {len(tgt_grads)} non-empty gradients!"

        optimizer.step()
        model.update_target_encoder()
        step_losses.append(loss.item())

    # Verify EMA updated target encoder
    new_tgt_param = next(model.target_encoder.parameters())
    param_diff = (sample_tgt_param - new_tgt_param).abs().sum().item()
    assert param_diff > 0, "EMA failed to update target encoder parameters!"

    duration = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

    # Save pilot checkpoint
    pilot_ckpt = project_root / cfg["experiment"]["output_dir"] / "checkpoints" / "checkpoint_pilot.pt"
    save_checkpoint(pilot_ckpt, model, optimizer=optimizer, epoch=0, step=len(step_losses), val_loss=step_losses[-1], config=cfg)

    print(f"Pilot completed in {duration:.2f}s | Steps: {len(step_losses)} | Final loss: {step_losses[-1]:.4f}")
    print(f"Peak VRAM: {peak_vram:.1f} MB | EMA delta: {param_diff:.6e}")
    print(f"Pilot Checkpoint saved to: {pilot_ckpt}")
    print("Pilot verification status: PASSED ALL CHECKS.")

    return {
        "pilot_passed": True,
        "steps_run": len(step_losses),
        "initial_loss": step_losses[0],
        "final_loss": step_losses[-1],
        "duration_sec": duration,
        "peak_vram_mb": peak_vram,
        "ema_param_diff": param_diff,
        "checkpoint_path": str(pilot_ckpt),
    }


def train_phase1(
    cfg: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    """Executes the full Phase 1 JEPA training run."""
    print("=" * 60)
    print("        STARTING FULL PHASE 1 JEPA PRETRAINING         ")
    print("=" * 60)

    device_str = "cuda" if torch.cuda.is_available() and cfg["experiment"].get("device", "cuda") == "cuda" else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    exp_cfg = cfg["experiment"]

    # Mixed precision setup
    use_autocast = (device.type == "cuda") and (train_cfg.get("mixed_precision", "bf16") in ["bf16", "fp16"])
    amp_dtype = torch.bfloat16 if train_cfg.get("mixed_precision") == "bf16" else torch.float16

    # 1. Load data
    train_parquet = project_root / data_cfg["train_parquet"]
    records = load_and_clean_states(train_parquet, min_word_count=data_cfg.get("min_words_total", 8))

    max_samples = data_cfg.get("max_train_samples")
    if max_samples and max_samples < len(records):
        records = records[:max_samples]
        print(f"Subsampled dataset to {len(records)} records.")

    train_recs, val_recs, split_meta = split_states(
        records,
        val_ratio=data_cfg.get("val_ratio", 0.10),
        seed=exp_cfg["seed"],
        deduplicate_identical_states=data_cfg.get("deduplicate_states", True),
    )
    print(f"Data Split: {len(train_recs):,} training states, {len(val_recs):,} validation states.")

    # 2. View Generation
    view_gen = ViewGenerator(
        strategy=data_cfg.get("strategy", "span_mask"),
        mask_ratio=data_cfg.get("mask_ratio", 0.25),
        min_words_total=data_cfg.get("min_words_total", 8),
        min_context_words=data_cfg.get("min_context_words", 3),
        min_target_words=data_cfg.get("min_target_words", 3),
        context_ratio=data_cfg.get("context_ratio", 0.60),
        seed=exp_cfg["seed"],
    )
    train_pairs, train_stats = view_gen.generate_pairs_batch(train_recs)
    val_pairs, val_stats = view_gen.generate_pairs_batch(val_recs)

    print(f"Generated {len(train_pairs):,} train pairs (rejected: {train_stats.rejected_states_count:,})")
    print(f"Generated {len(val_pairs):,} val pairs (rejected: {val_stats.rejected_states_count:,})")
    print(f"Avg words: Context={train_stats.avg_context_word_count:.1f}, Target={train_stats.avg_target_word_count:.1f}")

    # 3. Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone_name"])
    collator = JEPACollator(tokenizer, max_seq_length=train_cfg["max_seq_length"])

    num_workers = data_cfg.get("num_workers", 2)
    train_loader = DataLoader(
        JEPADataset(train_pairs),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        JEPADataset(val_pairs),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Model & Loss
    model = TextJEPA(
        backbone_name_or_path=model_cfg["backbone_name"],
        latent_dim=model_cfg["latent_dim"],
        predictor_hidden_dim=model_cfg["predictor_hidden_dim"],
        predictor_num_layers=model_cfg["predictor_num_layers"],
        pooling_type=model_cfg["pooling_type"],
        normalize_latents=model_cfg["normalize_latents"],
        ema_momentum=model_cfg["ema_momentum"],
        dropout_prob=model_cfg.get("dropout", 0.1),
    ).to(device)

    loss_fn = JEPALoss(
        loss_type=loss_cfg.get("loss_type", "normalized_mse"),
        variance_threshold=loss_cfg.get("variance_threshold", 0.01),
        similarity_threshold=loss_cfg.get("similarity_threshold", 0.95),
        min_mean_std_threshold=loss_cfg.get("min_mean_std_threshold", 0.05),
    )

    param_groups = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW([
        {"params": param_groups["encoder"], "lr": train_cfg["encoder_lr"]},
        {"params": param_groups["predictor"], "lr": train_cfg["predictor_lr"]},
    ], weight_decay=train_cfg["weight_decay"])

    total_steps = len(train_loader) * train_cfg["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # 5. Output Management & Tracking
    output_dir = project_root / exp_cfg["output_dir"]
    tracker = MetricsTracker(output_dir)
    checkpoints_dir = output_dir / "checkpoints"

    best_val_loss = float("inf")
    best_epoch = -1
    overall_start = time.time()

    print(f"Total Epochs: {train_cfg['num_epochs']} | Batches per epoch: {len(train_loader)} | Total steps: {total_steps}")
    print(f"Total Params: {model.count_parameters()['total_parameters']:,} (Trainable: {model.count_parameters()['trainable_parameters']:,})")

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        epoch_start = time.time()
        model.train()
        train_loss_acc = 0.0
        train_samples = 0

        for step, batch in enumerate(train_loader, 1):
            ctx_ids = batch["context_input_ids"].to(device, non_blocking=True)
            ctx_mask = batch["context_attention_mask"].to(device, non_blocking=True)
            tgt_ids = batch["target_input_ids"].to(device, non_blocking=True)
            tgt_mask = batch["target_attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                z_pred, z_target, z_context = model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
                loss = loss_fn(z_pred, z_target)

            loss.backward()
            if train_cfg.get("max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    list(param_groups["encoder"]) + list(param_groups["predictor"]),
                    train_cfg["max_grad_norm"],
                )

            optimizer.step()
            model.update_target_encoder()
            scheduler.step()

            bs = ctx_ids.size(0)
            train_loss_acc += loss.item() * bs
            train_samples += bs

            if step % train_cfg.get("logging_steps", 50) == 0 or step == len(train_loader):
                current_lr = optimizer.param_groups[0]["lr"]
                diag = loss_fn.compute_diagnostics(z_context)
                print(f"Epoch [{epoch}/{train_cfg['num_epochs']}] Step [{step}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | DimStd: {diag.mean_dimension_std:.3f} | Sim: {diag.mean_pairwise_similarity:.3f} | LR: {current_lr:.2e}")
                if diag.has_collapse_warning:
                    print(f"  [COLLAPSE WARNING] {diag.warning_reason}")

        train_epoch_loss = train_loss_acc / max(train_samples, 1)

        # Validation
        val_metrics = run_validation(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_autocast=use_autocast,
            dtype=amp_dtype,
        )
        epoch_time = time.time() - epoch_start
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

        print(f"--- Epoch {epoch} Summary: Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} "
              f"| Val Std: {val_metrics['val_mean_latent_std']:.3f} | Duration: {epoch_time:.1f}s | Peak VRAM: {peak_vram:.1f} MB ---")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "val_loss": val_metrics["val_loss"],
            "val_mean_latent_norm": val_metrics["val_mean_latent_norm"],
            "val_mean_latent_std": val_metrics["val_mean_latent_std"],
            "val_mean_pairwise_similarity": val_metrics["val_mean_pairwise_similarity"],
            "epoch_duration_sec": epoch_time,
            "peak_vram_mb": peak_vram,
        }
        tracker.record_epoch(epoch_record)

        # Save latest checkpoint
        save_checkpoint(
            checkpoint_path=checkpoints_dir / "checkpoint_latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=epoch * len(train_loader),
            val_loss=val_metrics["val_loss"],
            best_val_loss=min(best_val_loss, val_metrics["val_loss"]),
            config=cfg,
        )

        # Check for best checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            best_ckpt = checkpoints_dir / "checkpoint_best.pt"
            save_checkpoint(
                checkpoint_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=epoch * len(train_loader),
                val_loss=val_metrics["val_loss"],
                best_val_loss=best_val_loss,
                config=cfg,
            )
            # Export reusable encoder for Phase 2
            reusable_dir = checkpoints_dir / "reusable_encoder"
            save_reusable_encoder(reusable_dir, model, tokenizer_name=model_cfg["backbone_name"], config=cfg)
            print(f"  --> Saved new best checkpoint & reusable encoder (Val Loss: {best_val_loss:.4f})")

        # Save milestone checkpoints at epochs 1, 3, 5, 8, 10
        milestone_epochs = cfg.get("training", {}).get("checkpoint_epochs", [1, 3, 5, 8, 10])
        if epoch in milestone_epochs:
            milestone_ckpt = checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                checkpoint_path=milestone_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=epoch * len(train_loader),
                val_loss=val_metrics["val_loss"],
                best_val_loss=best_val_loss,
                config=cfg,
            )
            print(f"  --> Saved milestone checkpoint: {milestone_ckpt.name}")

    total_duration = time.time() - overall_start
    tracker.save_metrics()
    generated_plots = tracker.generate_plots()

    print("=" * 60)
    print(f"TRAINING COMPLETE in {total_duration:.1f}s | Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Plots saved to: {output_dir / 'plots'}")
    print(f"Reusable Encoder exported to: {checkpoints_dir / 'reusable_encoder'}")
    print("=" * 60)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_duration_sec": total_duration,
        "plots": [str(p) for p in generated_plots],
        "reusable_encoder_dir": str(checkpoints_dir / "reusable_encoder"),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1 JEPA Training")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config file")
    parser.add_argument("--pilot", action="store_true", help="Run pilot validation experiment only")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.pilot:
        run_pilot(cfg, project_root)
    else:
        train_phase1(cfg, project_root)


if __name__ == "__main__":
    main()
