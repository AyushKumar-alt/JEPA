"""
Predictor module for JEPA text representation learning.
Maps context latent representations into target representation space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictor(nn.Module):
    """
    Lightweight MLP predictor that predicts target latent representation
    from context latent representation.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 2,
        dropout_prob: float = 0.1,
        use_residual: bool = True,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_residual = use_residual and (latent_dim == hidden_dim)
        self.normalize_output = normalize_output

        layers = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, latent_dim))
        layers.append(nn.LayerNorm(latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, context_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_latent: (B, latent_dim)
        Returns:
            predicted_target: (B, latent_dim)
        """
        residual = context_latent
        out = self.net(context_latent)
        if self.use_residual:
            out = out + residual

        if self.normalize_output:
            return F.normalize(out, p=2, dim=-1)
        return out
