"""
Unit tests for full checkpointing and reusable encoder persistence.
"""

from pathlib import Path
import tempfile
import torch
import pytest

from src.jepa_model import TextJEPA
from src.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_reusable_encoder,
    load_reusable_encoder,
)


def test_checkpoint_save_and_reload():
    model = TextJEPA(
        backbone_name_or_path="distilbert-base-uncased",
        latent_dim=768,
        predictor_hidden_dim=768,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=2,
            step=100,
            val_loss=0.45,
            config={"test": True},
        )

        assert ckpt_path.exists()

        # Create fresh model and load
        new_model = TextJEPA(
            backbone_name_or_path="distilbert-base-uncased",
            latent_dim=768,
            predictor_hidden_dim=768,
        )
        new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

        meta = load_checkpoint(ckpt_path, new_model, new_opt)

        assert meta["epoch"] == 2
        assert meta["step"] == 100
        assert meta["val_loss"] == 0.45

        # Check parameter equality
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)


def test_reusable_encoder_export_and_inference():
    model = TextJEPA(
        backbone_name_or_path="distilbert-base-uncased",
        latent_dim=768,
        predictor_hidden_dim=768,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / "reusable_encoder"
        save_reusable_encoder(export_dir, model, tokenizer_name="distilbert-base-uncased")

        assert (export_dir / "context_encoder.pt").exists()
        assert (export_dir / "encoder_config.json").exists()

        # Load standalone encoder
        encoder, tokenizer, config = load_reusable_encoder(export_dir)
        sample_text = ["This is a test sentence for representation learning."]
        tokens = tokenizer(sample_text, return_tensors="pt")

        with torch.no_grad():
            latents = encoder(tokens["input_ids"], tokens["attention_mask"])

        assert latents.shape == (1, 768)
        assert torch.isfinite(latents).all()
