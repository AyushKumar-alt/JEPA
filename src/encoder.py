"""
Encoder module for JEPA text representation learning.
Implements mask-aware mean pooling and projection head for latent extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Any


def masked_mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes mask-aware mean pooling over token embeddings.
    Strictly ignores padding tokens so that padded representations are invariant
    to sequence padding length.

    Args:
        token_embeddings: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len) with 1 for tokens and 0 for padding.

    Returns:
        pooled: (batch_size, hidden_dim)
    """
    # Expand attention mask to match embedding dimensions: (B, L, 1)
    mask_expanded = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder with mask-aware pooling and projection head.
    Can be used as either the Context Encoder or Target Encoder.
    """

    def __init__(
        self,
        backbone_name_or_path: str = "distilbert-base-uncased",
        latent_dim: int = 768,
        pooling_type: str = "masked_mean",
        normalize_latents: bool = True,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name_or_path
        self.pooling_type = pooling_type
        self.normalize_latents = normalize_latents
        self.latent_dim = latent_dim

        self.backbone = AutoModel.from_pretrained(backbone_name_or_path)
        hidden_dim = self.backbone.config.hidden_size

        if hidden_dim != latent_dim:
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.Dropout(dropout_prob),
            )
        else:
            self.projection = nn.Identity()

        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encodes token sequence into fixed-dimensional latent representation.

        Args:
            input_ids: (B, L)
            attention_mask: (B, L)

        Returns:
            latents: (B, latent_dim)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = outputs.last_hidden_state  # (B, L, hidden_dim)

        if self.pooling_type == "masked_mean":
            pooled = masked_mean_pooling(token_embeddings, attention_mask)
        elif self.pooling_type == "cls":
            pooled = token_embeddings[:, 0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        projected = self.projection(pooled)
        normalized = self.layer_norm(projected)

        if self.normalize_latents:
            return F.normalize(normalized, p=2, dim=-1)
        return normalized

    def get_latent_dim(self) -> int:
        return self.latent_dim
