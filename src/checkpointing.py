"""
Checkpointing and model persistence module for Phase 1 JEPA training.
Saves full training checkpoints and exports standalone reusable context encoders
for downstream Phase 2 structured decision-making experiments.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import torch
from transformers import AutoTokenizer

from src.jepa_model import TextJEPA
from src.encoder import TextEncoder


def save_checkpoint(
    checkpoint_path: Path,
    model: TextJEPA,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    val_loss: float = float("inf"),
    best_val_loss: float = float("inf"),
    config: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves a complete training checkpoint with optimizer and scheduler state."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config or {},
        "metadata": extra_metadata or {},
    }

    torch.save(state, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: TextJEPA,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads a full training checkpoint and restores model, optimizer, and scheduler."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "val_loss": checkpoint.get("val_loss", float("inf")),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "config": checkpoint.get("config", {}),
        "metadata": checkpoint.get("metadata", {}),
    }


def save_reusable_encoder(
    export_dir: Path,
    model: TextJEPA,
    tokenizer_name: str = "distilbert-base-uncased",
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Exports ONLY the trained context encoder, its configuration, and its tokenizer.
    This creates an independent, lightweight artifact ready for downstream Phase 2 tasks.
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save context encoder state dict
    weights_path = export_dir / "context_encoder.pt"
    torch.save(model.context_encoder.state_dict(), weights_path)

    # 2. Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(str(export_dir))

    # 3. Save encoder metadata
    encoder_metadata = {
        "backbone_name": model.context_encoder.backbone_name,
        "latent_dim": model.context_encoder.latent_dim,
        "pooling_type": model.context_encoder.pooling_type,
        "normalize_latents": model.context_encoder.normalize_latents,
        "tokenizer_name": tokenizer_name,
        "full_config": config or {},
    }
    with open(export_dir / "encoder_config.json", "w") as f:
        json.dump(encoder_metadata, f, indent=2)

    return export_dir


def load_reusable_encoder(
    export_dir: Path,
    device: str = "cpu",
) -> Tuple[TextEncoder, Any, Dict[str, Any]]:
    """
    Loads a saved reusable encoder, tokenizer, and config.
    Returns:
        (encoder, tokenizer, config)
    """
    with open(export_dir / "encoder_config.json", "r") as f:
        meta = json.load(f)

    encoder = TextEncoder(
        backbone_name_or_path=meta["backbone_name"],
        latent_dim=meta["latent_dim"],
        pooling_type=meta["pooling_type"],
        normalize_latents=meta["normalize_latents"],
    )

    weights_path = export_dir / "context_encoder.pt"
    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(export_dir))
    return encoder, tokenizer, meta
