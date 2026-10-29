"""
Complete JEPA model module uniting Context Encoder, Target Encoder,
Predictor, and Exponential Moving Average (EMA) parameter updates.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from src.encoder import TextEncoder
from src.predictor import LatentPredictor


class TextJEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA) for text representations.
    Consists of:
      1. Context Encoder (learnable)
      2. Target Encoder (EMA-updated, frozen for optimizer gradients)
      3. Predictor (learnable)
    """

    def __init__(
        self,
        backbone_name_or_path: str = "distilbert-base-uncased",
        latent_dim: int = 768,
        predictor_hidden_dim: int = 768,
        predictor_num_layers: int = 2,
        pooling_type: str = "masked_mean",
        normalize_latents: bool = True,
        ema_momentum: float = 0.996,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum
        self.normalize_latents = normalize_latents

        # 1. Context Encoder
        self.context_encoder = TextEncoder(
            backbone_name_or_path=backbone_name_or_path,
            latent_dim=latent_dim,
            pooling_type=pooling_type,
            normalize_latents=normalize_latents,
            dropout_prob=dropout_prob,
        )

        # 2. Target Encoder (initialized with identical weights, frozen)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # 3. Predictor
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            dropout_prob=dropout_prob,
            normalize_output=normalize_latents,
        )

    def forward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a forward pass:
          1. Context encoder extracts context latent z_context
          2. Predictor maps z_context to predicted target latent z_pred
          3. Target encoder extracts ground-truth target latent z_target (detached from graph)

        Returns:
            z_pred: (B, D) predicted target latent (has grad)
            z_target: (B, D) target latent from target encoder (no grad)
            z_context: (B, D) context latent from context encoder (has grad)
        """
        # Encode context
        z_context = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
        )

        # Predict target latent
        z_pred = self.predictor(z_context)

        # Encode target with target encoder (NO GRADIENTS)
        with torch.no_grad():
            z_target = self.target_encoder(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
            )
            # Ensure target is detached
            z_target = z_target.detach()

        return z_pred, z_target, z_context

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        """
        Performs Exponential Moving Average (EMA) update on target encoder:
        theta_target = m * theta_target + (1 - m) * theta_context
        """
        m = momentum if momentum is not None else self.ema_momentum
        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_param.data.mul_(m).add_(ctx_param.data, alpha=1.0 - m)

    def get_trainable_parameters(self) -> Dict[str, list]:
        """
        Separates parameters into context encoder and predictor groups
        for differential learning rate scheduling.
        """
        return {
            "encoder": list(self.context_encoder.parameters()),
            "predictor": list(self.predictor.parameters()),
        }

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter count statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.context_encoder.parameters())
        predictor_params = sum(p.numel() for p in self.predictor.parameters())
        target_params = sum(p.numel() for p in self.target_encoder.parameters())

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "context_encoder_parameters": encoder_params,
            "predictor_parameters": predictor_params,
            "target_encoder_parameters": target_params,
        }
