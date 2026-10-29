"""
Loss functions and representation-collapse diagnostic tools for JEPA training.
Includes geometrically calibrated collapse thresholds and fixed-sample diagnostics.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CollapseDiagnostics:
    mean_latent_norm: float
    mean_dimension_std: float
    min_dimension_std: float
    max_dimension_std: float
    collapsed_dimension_fraction: float
    mean_pairwise_similarity: float
    has_collapse_warning: bool
    warning_reason: str


class JEPALoss(nn.Module):
    """
    Latent prediction loss for JEPA with collapse monitoring.
    Supports normalized MSE, cosine distance, standard MSE, and Smooth L1.
    """

    def __init__(
        self,
        loss_type: str = "normalized_mse",
        variance_threshold: float = 0.001,
        similarity_threshold: float = 0.90,
        min_mean_std_threshold: float = 0.012,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.variance_threshold = variance_threshold
        self.similarity_threshold = similarity_threshold
        self.min_mean_std_threshold = min_mean_std_threshold

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the latent regression loss.
        Target is expected to be detached.
        """
        if self.loss_type == "normalized_mse":
            u_pred = F.normalize(z_pred, p=2, dim=-1)
            u_target = F.normalize(z_target, p=2, dim=-1)
            return F.mse_loss(u_pred, u_target)
        elif self.loss_type == "cosine":
            u_pred = F.normalize(z_pred, p=2, dim=-1)
            u_target = F.normalize(z_target, p=2, dim=-1)
            return 1.0 - (u_pred * u_target).sum(dim=-1).mean()
        elif self.loss_type == "mse":
            return F.mse_loss(z_pred, z_target)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(z_pred, z_target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    @torch.no_grad()
    def compute_diagnostics(self, latents: torch.Tensor) -> CollapseDiagnostics:
        """
        Computes representation diversity and collapse indicators across a batch.
        Calibrated for 768-dimensional embeddings on the unit sphere.
        """
        b_size, dim = latents.shape
        if b_size < 2:
            return CollapseDiagnostics(
                mean_latent_norm=float(latents.norm(dim=-1).mean().item()),
                mean_dimension_std=1.0,
                min_dimension_std=1.0,
                max_dimension_std=1.0,
                collapsed_dimension_fraction=0.0,
                mean_pairwise_similarity=0.0,
                has_collapse_warning=False,
                warning_reason="Batch size < 2; diagnostics skipped.",
            )

        # 1. Latent norms
        norms = latents.norm(dim=-1)
        mean_norm = float(norms.mean().item())

        # 2. Per-dimension standard deviation across the batch
        dim_std = torch.std(latents, dim=0, unbiased=False)
        mean_std = float(dim_std.mean().item())
        min_std = float(dim_std.min().item())
        max_std = float(dim_std.max().item())

        collapsed_dims = (dim_std < self.variance_threshold).sum().item()
        collapsed_fraction = float(collapsed_dims / dim)

        # 3. Pairwise cosine similarity across the batch
        u_latents = F.normalize(latents, p=2, dim=-1)
        sim_matrix = torch.matmul(u_latents, u_latents.T)
        mask = ~torch.eye(b_size, dtype=torch.bool, device=latents.device)
        mean_pairwise_sim = float(sim_matrix[mask].mean().item())

        # 4. Collapse indicators
        warnings = []
        if mean_std < self.min_mean_std_threshold:
            warnings.append(f"Mean per-dim std ({mean_std:.4f}) < threshold ({self.min_mean_std_threshold})")
        if collapsed_fraction > 0.30:
            warnings.append(f"{collapsed_fraction * 100:.1f}% dimensions have variance < {self.variance_threshold}")
        if mean_pairwise_sim > self.similarity_threshold:
            warnings.append(f"Mean pairwise cosine similarity ({mean_pairwise_sim:.4f}) > {self.similarity_threshold}")

        has_warning = len(warnings) > 0
        reason = "; ".join(warnings) if has_warning else "Healthy representation diversity"

        return CollapseDiagnostics(
            mean_latent_norm=mean_norm,
            mean_dimension_std=mean_std,
            min_dimension_std=min_std,
            max_dimension_std=max_std,
            collapsed_dimension_fraction=collapsed_fraction,
            mean_pairwise_similarity=mean_pairwise_sim,
            has_collapse_warning=has_warning,
            warning_reason=reason,
        )
