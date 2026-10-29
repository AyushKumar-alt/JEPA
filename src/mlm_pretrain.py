"""
Masked Language Model (MLM) Pretraining Baseline (Model C).
Pretrains DistilBert on the exact same label-free state corpus as JEPA
using standard 15% token masking to ensure a fair, matched baseline.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
)

import sys
project_root_dir = Path(__file__).resolve().parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

from src.data_processing import load_and_clean_states, split_states
from src.encoder import TextEncoder
from src.checkpointing import save_reusable_encoder


class MLMTextDataset(Dataset):
    """Dataset yielding state text for MLM pretraining."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.texts = [r["state_text"] for r in records]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def train_mlm(
    cfg: Dict[str, Any],
    project_root: Path,
    num_epochs: int = 5,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    print("=" * 60)
    print("   STARTING MASKED LANGUAGE MODEL (MLM) PRETRAINING (MODEL C)   ")
    print("=" * 60)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    data_cfg = cfg["data"]
    train_parquet = project_root / data_cfg["train_parquet"]
    records = load_and_clean_states(train_parquet, min_word_count=8)

    max_samples = data_cfg.get("max_train_samples")
    if max_samples and max_samples < len(records):
        records = records[:max_samples]

    train_recs, val_recs, _ = split_states(
        records,
        val_ratio=data_cfg.get("val_ratio", 0.10),
        seed=cfg["experiment"]["seed"],
        deduplicate_identical_states=True,
    )
    print(f"MLM Data Split: {len(train_recs):,} training states, {len(val_recs):,} validation states.")

    backbone_name = cfg["model"]["backbone_name"]
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    model = DistilBertForMaskedLM.from_pretrained(backbone_name).to(device)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    max_seq_len = cfg["training"].get("max_seq_length", 256)
    batch_size = cfg["training"].get("batch_size", 32)

    def collate_fn(batch_texts: List[str]):
        tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        return data_collator(tokenized["input_ids"])

    train_loader = DataLoader(
        MLMTextDataset(train_recs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        MLMTextDataset(val_recs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs if not max_steps else max_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    output_dir = project_root / "outputs" / "mlm"
    checkpoints_dir = output_dir / "checkpoints"
    reusable_dir = checkpoints_dir / "reusable_encoder"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reusable_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    overall_start = time.time()
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_acc = 0.0
        train_tokens = 0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            train_loss_acc += loss.item() * batch["input_ids"].size(0)
            train_tokens += batch["input_ids"].size(0)

            if step % 100 == 0 or step == len(train_loader):
                print(f"MLM Epoch [{epoch}/{num_epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

            if max_steps and global_step >= max_steps:
                break

        train_loss = train_loss_acc / max(train_tokens, 1)

        # Validation
        model.eval()
        val_loss_acc = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    outputs = model(**batch)
                    val_loss_acc += outputs.loss.item() * batch["input_ids"].size(0)
                    val_tokens += batch["input_ids"].size(0)

        val_loss = val_loss_acc / max(val_tokens, 1)
        epoch_time = time.time() - epoch_start
        print(f"--- MLM Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s ---")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save underlying DistilBERT transformer encoder weights into TextEncoder format
            base_transformer = model.distilbert
            text_encoder = TextEncoder(
                backbone_name_or_path=backbone_name,
                latent_dim=768,
                pooling_type="masked_mean",
                normalize_latents=True,
            )
            text_encoder.backbone.load_state_dict(base_transformer.state_dict())
            torch.save(text_encoder.state_dict(), reusable_dir / "context_encoder.pt")
            tokenizer.save_pretrained(str(reusable_dir))

            import json
            with open(reusable_dir / "encoder_config.json", "w") as f:
                json.dump({
                    "backbone_name": backbone_name,
                    "latent_dim": 768,
                    "pooling_type": "masked_mean",
                    "normalize_latents": True,
                    "model_type": "mlm_pretrained",
                    "best_val_loss": best_val_loss,
                }, f, indent=2)
            print(f"  --> Exported best MLM reusable encoder to {reusable_dir}")

        if max_steps and global_step >= max_steps:
            break

    total_time = time.time() - overall_start
    print(f"MLM Training completed in {total_time:.1f}s | Best Val Loss: {best_val_loss:.4f}")
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_time_sec": total_time,
        "reusable_encoder_dir": str(reusable_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_mlm(cfg, project_root, num_epochs=args.epochs, max_steps=args.max_steps)
