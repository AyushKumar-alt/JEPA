"""
Unit tests for mini training steps, finite loss, and validation invariance.
"""

import torch
import pytest
from src.jepa_model import TextJEPA
from src.losses import JEPALoss


def test_mini_training_step():
    model = TextJEPA(
        backbone_name_or_path="distilbert-base-uncased",
        latent_dim=768,
        predictor_hidden_dim=768,
        ema_momentum=0.99,
    )
    model.train()
    loss_fn = JEPALoss(loss_type="normalized_mse")

    param_groups = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW([
        {"params": param_groups["encoder"], "lr": 1e-4},
        {"params": param_groups["predictor"], "lr": 1e-4},
    ])

    ctx_ids = torch.randint(0, 1000, (4, 16))
    ctx_mask = torch.ones(4, 16)
    tgt_ids = torch.randint(0, 1000, (4, 16))
    tgt_mask = torch.ones(4, 16)

    optimizer.zero_grad()
    z_pred, z_target, z_context = model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
    loss = loss_fn(z_pred, z_target)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()
    optimizer.step()
    model.update_target_encoder()


def test_validation_does_not_alter_parameters():
    model = TextJEPA(
        backbone_name_or_path="distilbert-base-uncased",
        latent_dim=768,
        predictor_hidden_dim=768,
    )
    model.eval()

    # Clone all parameters
    initial_params = [p.clone().detach() for p in model.parameters()]

    ctx_ids = torch.randint(0, 1000, (2, 16))
    ctx_mask = torch.ones(2, 16)
    tgt_ids = torch.randint(0, 1000, (2, 16))
    tgt_mask = torch.ones(2, 16)

    with torch.no_grad():
        z_pred, z_target, z_context = model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
        loss = JEPALoss()(z_pred, z_target)

    # Verify no parameter changed
    for p_init, p_now in zip(initial_params, model.parameters()):
        assert torch.equal(p_init, p_now), "Model parameters were modified during validation!"
