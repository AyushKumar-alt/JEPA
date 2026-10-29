"""
Unit tests for TextEncoder and mask-aware pooling.
"""

import torch
import pytest
from src.encoder import TextEncoder, masked_mean_pooling


def test_masked_mean_pooling_padding_invariance():
    """
    CRITICAL TEST: Verifies that padding tokens have ZERO influence
    on the pooled representation.
    """
    hidden_dim = 16
    unpadded_len = 4
    padded_len = 10

    torch.manual_seed(42)
    # Token embeddings for real tokens
    real_tokens = torch.randn(1, unpadded_len, hidden_dim)
    unpadded_mask = torch.ones(1, unpadded_len)

    pooled_unpadded = masked_mean_pooling(real_tokens, unpadded_mask)

    # Pad with arbitrary noise in the padded positions
    pad_tokens = torch.randn(1, padded_len - unpadded_len, hidden_dim) * 100.0  # huge noise
    padded_tokens = torch.cat([real_tokens, pad_tokens], dim=1)
    padded_mask = torch.cat([torch.ones(1, unpadded_len), torch.zeros(1, padded_len - unpadded_len)], dim=1)

    pooled_padded = masked_mean_pooling(padded_tokens, padded_mask)

    assert torch.allclose(pooled_unpadded, pooled_padded, atol=1e-6), (
        f"Padding invariance violated! Diff: {(pooled_unpadded - pooled_padded).abs().max().item()}"
    )


def test_text_encoder_output_dimensions():
    encoder = TextEncoder(
        backbone_name_or_path="distilbert-base-uncased",
        latent_dim=768,
        pooling_type="masked_mean",
        normalize_latents=True,
    )
    encoder.eval()

    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones(2, 16)

    with torch.no_grad():
        latents = encoder(input_ids, attention_mask)

    assert latents.shape == (2, 768)
    # Check L2 normalization (norm should be ~1.0)
    norms = latents.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
