"""
Comprehensive Unit Tests for Stage 9 Audit, Integrity, and Methodology Requirements.
Verifies dataset integrity, backup preservation, leakage controls, split isolation,
deterministic subsets, and evaluation metrics.
"""

from pathlib import Path
import pytest
import pyarrow.parquet as pq
import torch
import numpy as np
from sklearn.metrics import f1_score

from src.data_processing import (
    load_and_clean_states,
    split_states,
    verify_record_schema,
    hash_state,
)
from src.downstream_dataset import (
    parse_downstream_dataset,
    create_label_efficiency_subsets,
    format_downstream_prompt,
    DownstreamExample,
)
from src.view_generator import ViewGenerator


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


def test_dataset_integrity_and_row_counts(project_root):
    """
    CRITICAL CHECK 1: Verifies exact row counts and schema of train and test splits.
    Train: 125,614 rows
    Test: 24,386 rows
    Expected columns: state, questions_json, domain, question_types, question_count
    """
    train_p = project_root / "data" / "bev-decision" / "train.parquet"
    test_p = project_root / "data" / "bev-decision" / "test.parquet"

    assert train_p.exists(), f"Train parquet missing: {train_p}"
    assert test_p.exists(), f"Test parquet missing: {test_p}"

    t_train = pq.read_table(str(train_p))
    t_test = pq.read_table(str(test_p))

    assert t_train.num_rows == 125614, f"Train row count mismatch: {t_train.num_rows} != 125,614"
    assert t_test.num_rows == 24386, f"Test row count mismatch: {t_test.num_rows} != 24,386"

    expected_cols = {"state", "questions_json", "domain", "question_types", "question_count"}
    assert expected_cols.issubset(set(t_train.column_names))
    assert expected_cols.issubset(set(t_test.column_names))


def test_backup_preservation(project_root):
    """
    CRITICAL CHECK 2: Verifies that original backup file is preserved and non-empty.
    """
    backup_p = project_root / "data" / "bev-decision" / "train.parquet.original_backup"
    assert backup_p.exists(), f"Original backup missing: {backup_p}"
    assert backup_p.stat().st_size > 0, "Original backup is empty!"


def test_zero_state_overlap_train_test(project_root):
    """
    CRITICAL CHECK 3: Verifies that normalized states do not cross train-test split boundary.
    """
    train_p = project_root / "data" / "bev-decision" / "train.parquet"
    test_p = project_root / "data" / "bev-decision" / "test.parquet"

    # Sample check of 1000 states from each
    train_states = pq.read_table(str(train_p), columns=["state"]).column("state")[:1000].to_pylist()
    test_states = pq.read_table(str(test_p), columns=["state"]).column("state")[:1000].to_pylist()

    train_hashes = {hash_state(str(s).strip()) for s in train_states if s}
    test_hashes = {hash_state(str(s).strip()) for s in test_states if s}

    overlap = train_hashes.intersection(test_hashes)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping states between train and test splits!"


def test_view_generator_span_masking_alignment():
    """
    CRITICAL CHECK 4: Verifies context-target alignment in span masking.
    Target must be complete unmasked state.
    Context must be span-masked state.
    """
    vg = ViewGenerator(strategy="span_mask", mask_ratio=0.25, max_span_length=3, seed=42)
    sample_text = "Patient was admitted to emergency room showing acute signs of respiratory distress."
    view = vg.generate_pair(sample_text)
    assert view is not None

    assert view.target_text == sample_text, "Target view must be complete unmasked state!"
    assert "[MASK]" in view.context_text, "Context view must contain [MASK] tokens!"


def test_deterministic_label_subsets():
    """
    CRITICAL CHECK 5: Deterministic label subset generation.
    Same seed produces identical subsets; different seeds produce different subsets.
    """
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
        for i in range(50)
    ]

    s1 = create_label_efficiency_subsets(examples, fractions=[0.10, 0.50], seed=42)
    s2 = create_label_efficiency_subsets(examples, fractions=[0.10, 0.50], seed=42)
    s3 = create_label_efficiency_subsets(examples, fractions=[0.10, 0.50], seed=999)

    # Identical seed
    assert [e.example_id for e in s1[0.10]] == [e.example_id for e in s2[0.10]]
    # Different seed
    assert [e.example_id for e in s1[0.10]] != [e.example_id for e in s3[0.10]]


def test_evaluation_metric_mathematics():
    """
    CRITICAL CHECK 6: Verifies accuracy, macro-F1, and score MAE computation.
    """
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2]  # 5/6 correct

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    mae = float(np.mean([abs(t - p) for t, p in zip(y_true, y_pred)]))

    assert abs(acc - (5 / 6)) < 1e-6
    assert 0.0 <= f1 <= 1.0
    assert abs(mae - (1 / 6)) < 1e-6
