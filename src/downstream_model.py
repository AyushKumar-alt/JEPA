"""
Downstream Decision Prediction Architecture for Structured Decision-Making.
Supports Model A (Supervised DistilBERT baseline), Model B (JEPA-pretrained),
and Model C (MLM-pretrained), sharing identical multi-task decision heads.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import TextEncoder
from src.checkpointing import load_reusable_encoder


class DecisionPredictionModel(nn.Module):
    """
    Downstream Decision Model uniting a representation encoder with
    task-specific decision heads for CHOICE, NOUL, and SCORE tasks.
    """

    def __init__(
        self,
        model_type: str = "supervised",
        pretrained_encoder_dir: Optional[Path] = None,
        backbone_name: str = "distilbert-base-uncased",
        latent_dim: int = 768,
        max_choice_classes: int = 10,
        max_score_classes: int = 7,
        dropout_prob: float = 0.10,
    ):
        super().__init__()
        self.model_type = model_type
        self.latent_dim = latent_dim

        # 1. Base Representation Encoder
        if model_type in ["jepa", "mlm"] and pretrained_encoder_dir and (pretrained_encoder_dir / "context_encoder.pt").exists():
            print(f"Loading {model_type.upper()} pretrained encoder from: {pretrained_encoder_dir}")
            self.encoder, _, _ = load_reusable_encoder(pretrained_encoder_dir)
        else:
            print(f"Initializing standard supervised encoder from {backbone_name}")
            self.encoder = TextEncoder(
                backbone_name_or_path=backbone_name,
                latent_dim=latent_dim,
                pooling_type="masked_mean",
                normalize_latents=True,
            )

        # 2. Decision Heads
        self.dropout = nn.Dropout(dropout_prob)
        self.choice_head = nn.Linear(latent_dim, max_choice_classes)
        self.noul_head = nn.Linear(latent_dim, 2)
        self.score_head = nn.Linear(latent_dim, max_score_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_types: List[str],
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Processes batch of mixed or single task instances.
        Returns logits, predictions, and cross-entropy loss if targets provided.
        """
        latents = self.encoder(input_ids, attention_mask)
        latents = self.dropout(latents)

        batch_size = input_ids.size(0)
        predictions = [0] * batch_size
        losses = []

        # Vectorized routing by task type for maximal GPU throughput
        from collections import defaultdict
        indices_by_type = defaultdict(list)
        for i, t in enumerate(task_types):
            indices_by_type[t].append(i)

        for t_type, idx_list in indices_by_type.items():
            idx_tensor = torch.tensor(idx_list, device=latents.device)
            sub_latents = latents[idx_tensor]
            if t_type == "choice":
                head = self.choice_head
            elif t_type == "noul":
                head = self.noul_head
            elif t_type == "score":
                head = self.score_head
            else:
                head = self.choice_head

            sub_logits = head(sub_latents)
            sub_preds = torch.argmax(sub_logits, dim=-1).cpu().tolist()
            for orig_idx, p in zip(idx_list, sub_preds):
                predictions[orig_idx] = p

            if targets is not None:
                sub_targets = targets[idx_tensor]
                sub_loss = self.loss_fn(sub_logits, sub_targets)
                losses.append(sub_loss * len(idx_list))

        total_loss = (sum(losses) / batch_size) if losses else None

        return {
            "predictions": predictions,
            "loss": total_loss,
            "latents": latents,
        }
