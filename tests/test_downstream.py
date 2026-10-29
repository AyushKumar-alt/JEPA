"""
Unit tests for downstream dataset parser, decision model, and evaluation metrics.
"""

import pytest
import torch
from pathlib import Path

from src.downstream_dataset import (
    parse_downstream_dataset,
    format_downstream_prompt,
    create_label_efficiency_subsets,
    DownstreamExample,
)
from src.downstream_model import DecisionPredictionModel


def test_format_downstream_prompt_no_leakage():
    # 1. CHOICE
    prompt, opt_names, opt_to_idx = format_downstream_prompt(
        state_text="Customer reported a billing error.",
        instructions="What is the intent?",
        task_type="choice",
        criteria={"refund": "Request refund", "inquiry": "Billing inquiry"},
    )
    assert "Customer reported a billing error." in prompt
    assert "refund: Request refund" in prompt
    assert "Correct Answer" not in prompt
    assert opt_to_idx["refund"] == 0
    assert opt_to_idx["inquiry"] == 1

    # 2. NOUL
    prompt_noul, _, _ = format_downstream_prompt(
        state_text="Patient displays elevated heart rate.",
        instructions="Is medication recommended?",
        task_type="noul",
        criteria=None,
    )
    assert "Proposition:" in prompt_noul
    assert "Correct Answer" not in prompt_noul

    # 3. SCORE
    prompt_score, score_names, _ = format_downstream_prompt(
        state_text="Product review text.",
        instructions="Rate the sentiment.",
        task_type="score",
        criteria=["Bad", "Poor", "Neutral", "Good", "Great"],
    )
    assert "Score Scale:" in prompt_score
    assert len(score_names) == 5


def test_create_label_efficiency_subsets_nesting():
    examples = [
        DownstreamExample(
            example_id=f"ex_{i}",
            state_id=f"st_{i // 2}",
            state_hash=f"hash_{i // 2}",
            domain="test",
            task_type="choice",
            input_text="State: test Question: test",
            target_idx=0,
            num_classes=4,
            option_labels=["A", "B", "C", "D"],
        )
        for i in range(40)
    ]

    subsets = create_label_efficiency_subsets(examples, fractions=[0.10, 0.50, 1.00], seed=42)
    s10 = set(e.example_id for e in subsets[0.10])
    s50 = set(e.example_id for e in subsets[0.50])
    s100 = set(e.example_id for e in subsets[1.00])

    # Verify strict nesting: S10 is subset of S50, S50 is subset of S100
    assert s10.issubset(s50)
    assert s50.issubset(s100)
    assert len(s10) < len(s50) < len(s100)


def test_downstream_model_forward():
    model = DecisionPredictionModel(model_type="supervised")
    model.eval()

    input_ids = torch.randint(0, 1000, (3, 16))
    attention_mask = torch.ones(3, 16)
    task_types = ["choice", "noul", "score"]
    targets = torch.tensor([1, 0, 3], dtype=torch.long)

    out = model(input_ids, attention_mask, task_types, targets=targets)
    assert len(out["predictions"]) == 3
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])
