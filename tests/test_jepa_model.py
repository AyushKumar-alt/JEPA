"""
Unit tests for TextJEPA architecture, gradient isolation, and EMA update dynamics.
"""

import torch
import pytest
from src.jepa_model import TextJEPA
from src.losses import JEPALoss


@pytest.fixture
def jepa_model():
    return TextJEPA(
        backbone_name_or_path="distilbert-base-uncased",
        latent_dim=768,
        predictor_hidden_dim=768,
        ema_momentum=0.99,
        normalize_latents=True,
    )


def test_jepa_forward_shapes(jepa_model):
    jepa_model.eval()
    ctx_ids = torch.randint(0, 1000, (2, 12))
    ctx_mask = torch.ones(2, 12)
    tgt_ids = torch.randint(0, 1000, (2, 8))
    tgt_mask = torch.ones(2, 8)

    with torch.no_grad():
        z_pred, z_target, z_context = jepa_model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)

    assert z_pred.shape == (2, 768)
    assert z_target.shape == (2, 768)
    assert z_context.shape == (2, 768)


def test_gradient_isolation(jepa_model):
    """
    CRITICAL TEST:
    Verifies that gradients flow to context encoder and predictor,
    and ZERO gradients reach the target encoder.
    """
    jepa_model.train()
    loss_fn = JEPALoss(loss_type="normalized_mse")

    ctx_ids = torch.randint(0, 1000, (2, 12))
    ctx_mask = torch.ones(2, 12)
    tgt_ids = torch.randint(0, 1000, (2, 8))
    tgt_mask = torch.ones(2, 8)

    z_pred, z_target, z_context = jepa_model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
    loss = loss_fn(z_pred, z_target)
    loss.backward()

    # 1. Target encoder must have NO gradients
    for name, p in jepa_model.target_encoder.named_parameters():
        assert p.grad is None, f"Target encoder parameter {name} received ordinary gradient!"

    # 2. Predictor must have non-zero gradients
    pred_grads = [p.grad for p in jepa_model.predictor.parameters() if p.grad is not None]
    assert len(pred_grads) > 0 and sum(g.norm().item() for g in pred_grads) > 0

    # 3. Context encoder must have non-zero gradients
    ctx_grads = [p.grad for p in jepa_model.context_encoder.parameters() if p.grad is not None]
    assert len(ctx_grads) > 0 and sum(g.norm().item() for g in ctx_grads) > 0


def test_ema_mathematical_correctness(jepa_model):
    """
    CRITICAL TEST:
    Verifies that target parameter = momentum * target + (1 - momentum) * context.
    """
    momentum = 0.90
    # Artificially modify one context parameter
    ctx_first = next(jepa_model.context_encoder.parameters())
    tgt_first = next(jepa_model.target_encoder.parameters())

    ctx_orig = ctx_first.clone().detach()
    tgt_orig = tgt_first.clone().detach()

    # Update context with a deterministic offset
    with torch.no_grad():
        ctx_first.add_(1.0)

    # Perform EMA update
    jepa_model.update_target_encoder(momentum=momentum)

    # Expected value:
    expected_tgt = momentum * tgt_orig + (1.0 - momentum) * (ctx_orig + 1.0)
    actual_tgt = next(jepa_model.target_encoder.parameters()).data

    assert torch.allclose(actual_tgt, expected_tgt, atol=1e-6), "EMA update formula mismatch!"
