"""
Label-Efficiency Fine-Tuning Pipeline for Downstream Decision-Making.
Compares Model A (Supervised), Model B (JEPA), and Model C (MLM)
across label fractions (10%, 25%, 50%, 100%) and seeds.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import sys
project_root_dir = Path(__file__).resolve().parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

from src.downstream_dataset import (
    DownstreamExample,
    parse_downstream_dataset,
    create_label_efficiency_subsets,
)
from src.downstream_model import DecisionPredictionModel


class DecisionDataset(Dataset):
    def __init__(self, examples: List[DownstreamExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DownstreamExample:
        return self.examples[idx]


class DecisionCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_length: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[DownstreamExample]) -> Dict[str, Any]:
        texts = [ex.input_text for ex in batch]
        task_types = [ex.task_type for ex in batch]
        targets = torch.tensor([ex.target_idx for ex in batch], dtype=torch.long)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "task_types": task_types,
            "targets": targets,
            "domains": [ex.domain for ex in batch],
        }


def evaluate_downstream(
    model: DecisionPredictionModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Computes overall and per-task evaluation metrics on a validation set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_overall = 0

    per_task_correct = {"choice": 0, "noul": 0, "score": 0}
    per_task_total = {"choice": 0, "noul": 0, "score": 0}
    score_abs_errors = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            task_types = batch["task_types"]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(input_ids, attention_mask, task_types, targets=targets)

            bs = input_ids.size(0)
            total_loss += out["loss"].item() * bs
            total_samples += bs

            preds = out["predictions"]
            tgt_list = targets.cpu().tolist()

            for p, t, tt in zip(preds, tgt_list, task_types):
                is_corr = (p == t)
                if is_corr:
                    correct_overall += 1
                if tt in per_task_total:
                    per_task_total[tt] += 1
                    if is_corr:
                        per_task_correct[tt] += 1
                if tt == "score":
                    score_abs_errors.append(abs(p - t))

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct_overall / max(total_samples, 1)

    choice_acc = per_task_correct["choice"] / max(per_task_total["choice"], 1)
    noul_acc = per_task_correct["noul"] / max(per_task_total["noul"], 1)
    score_acc = per_task_correct["score"] / max(per_task_total["score"], 1)
    score_mae = sum(score_abs_errors) / max(len(score_abs_errors), 1)

    return {
        "val_loss": avg_loss,
        "val_accuracy": acc,
        "val_choice_acc": choice_acc,
        "val_noul_acc": noul_acc,
        "val_score_acc": score_acc,
        "val_score_mae": score_mae,
    }


def train_downstream_model(
    model_type: str,
    label_fraction: float,
    seed: int,
    project_root: Path,
    epochs: int = 5,
    patience: int = 2,
    lr: float = 3e-5,
    batch_size: int = 32,
    max_train_rows: Optional[int] = None,
) -> Dict[str, Any]:
    print("=" * 60)
    print(f"DOWNSTREAM TRAINING: Model={model_type.upper()} | Frac={label_fraction*100:.0f}% | Seed={seed}")
    print("=" * 60)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    train_parquet = project_root / "data" / "bev-decision" / "train.parquet"
    all_examples = parse_downstream_dataset(train_parquet)

    # Segregate unique states into 90% train candidate, 10% validation
    state_hashes = sorted(list(set(e.state_hash for e in all_examples)))
    import random
    rng = random.Random(42)  # fixed validation split
    rng.shuffle(state_hashes)

    val_hash_count = int(len(state_hashes) * 0.10)
    val_hashes = set(state_hashes[:val_hash_count])
    train_candidate_hashes = set(state_hashes[val_hash_count:])

    val_examples = [e for e in all_examples if e.state_hash in val_hashes]
    train_candidate_examples = [e for e in all_examples if e.state_hash in train_candidate_hashes]

    # Optional speedup cap for fast experimentation
    if max_train_rows and max_train_rows < len(train_candidate_examples):
        train_candidate_examples = train_candidate_examples[:max_train_rows]
        val_examples = val_examples[: int(max_train_rows * 0.10)]

    # Generate nested label efficiency subsets
    subsets = create_label_efficiency_subsets(
        train_candidate_examples,
        fractions=[0.10, 0.25, 0.50, 1.00],
        seed=seed,
    )
    active_train_examples = subsets[label_fraction]
    print(f"Active training instances: {len(active_train_examples):,} (Validation: {len(val_examples):,})")

    # Locate pretrained encoder
    encoder_dir = None
    if model_type == "jepa":
        encoder_dir = project_root / "outputs" / "phase1" / "checkpoints" / "reusable_encoder"
    elif model_type == "mlm":
        encoder_dir = project_root / "outputs" / "mlm" / "checkpoints" / "reusable_encoder"

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collator = DecisionCollator(tokenizer, max_seq_length=256)

    train_loader = DataLoader(
        DecisionDataset(active_train_examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    output_dir = project_root / "outputs" / "phase2"
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoints_dir / f"{model_type}_frac{int(label_fraction*100)}_seed{seed}_best.pt"
    metrics_log_path = metrics_dir / f"{model_type}_frac{int(label_fraction*100)}_seed{seed}_train_metrics.json"

    val_loader = DataLoader(
        DecisionDataset(val_examples),
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = DecisionPredictionModel(
        model_type=model_type,
        pretrained_encoder_dir=encoder_dir,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    start_time = time.time()
    epoch_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_samples = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            task_types = batch["task_types"]

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(input_ids, attention_mask, task_types, targets=targets)
                loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            bs = input_ids.size(0)
            train_loss_acc += loss.item() * bs
            train_samples += bs

        train_loss = train_loss_acc / max(train_samples, 1)

        val_metrics = evaluate_downstream(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']*100:.2f}% | Score MAE: {val_metrics['val_score_mae']:.3f}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["val_accuracy"],
            "val_choice_acc": val_metrics["val_choice_acc"],
            "val_noul_acc": val_metrics["val_noul_acc"],
            "val_score_acc": val_metrics["val_score_acc"],
            "val_score_mae": val_metrics["val_score_mae"],
        }
        epoch_history.append(epoch_record)

        # Check for improvement in val loss (or accuracy)
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_val_acc = val_metrics["val_accuracy"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_type": model_type,
                "label_fraction": label_fraction,
                "seed": seed,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "state_dict": model.state_dict(),
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    duration = time.time() - start_time
    print(f"Finished {model_type.upper()} frac={label_fraction*100:.0f}% in {duration:.1f}s | Best Val Acc: {best_val_acc*100:.2f}%")

    run_summary = {
        "model_type": model_type,
        "label_fraction": label_fraction,
        "seed": seed,
        "epochs_trained": len(epoch_history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "duration_sec": duration,
        "checkpoint_path": str(best_ckpt_path),
        "epoch_history": epoch_history,
    }

    with open(metrics_log_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Saved run metrics to {metrics_log_path}")

    return run_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=["supervised", "jepa", "mlm"], required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    train_downstream_model(
        model_type=args.model_type,
        label_fraction=args.fraction,
        seed=args.seed,
        project_root=project_root,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_rows=args.max_rows,
    )
