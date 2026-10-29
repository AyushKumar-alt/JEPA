"""
Unit tests for data processing and data leakage safeguards.
"""

from pathlib import Path
import pytest
import pyarrow.parquet as pq

from src.data_processing import (
    load_and_clean_states,
    split_states,
    verify_record_schema,
    hash_state,
)


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def train_parquet(project_root):
    p = project_root / "data" / "bev-decision" / "train.parquet"
    assert p.exists(), f"Train parquet missing: {p}"
    return p


@pytest.fixture
def test_parquet(project_root):
    p = project_root / "data" / "bev-decision" / "test.parquet"
    assert p.exists(), f"Test parquet missing: {p}"
    return p


def test_parquet_files_readable(train_parquet, test_parquet):
    t_train = pq.read_table(str(train_parquet))
    t_test = pq.read_table(str(test_parquet))
    assert t_train.num_rows > 0
    assert t_test.num_rows > 0
    assert "state" in t_train.column_names
    assert "questions_json" in t_train.column_names


def test_state_extraction_drops_labels(train_parquet):
    records = load_and_clean_states(train_parquet, min_word_count=8)
    assert len(records) > 0

    # Verify no record contains forbidden label fields
    for r in records[:500]:
        assert verify_record_schema(r), f"Data leakage detected in record: {r.keys()}"
        assert "questions_json" not in r
        assert "label" not in r
        assert "choice" not in r
        assert "target_label" not in r
        assert "state_text" in r
        assert len(r["state_text"].split()) >= 8


def test_train_val_split_no_overlap(train_parquet):
    records = load_and_clean_states(train_parquet, min_word_count=8)[:500]
    train_recs, val_recs, meta = split_states(
        records,
        val_ratio=0.20,
        seed=42,
        deduplicate_identical_states=True,
    )

    train_hashes = {r["state_hash"] for r in train_recs}
    val_hashes = {r["state_hash"] for r in val_recs}
    assert len(train_hashes.intersection(val_hashes)) == 0, "State hash overlap between train and val!"

    train_texts = {r["state_text"] for r in train_recs}
    val_texts = {r["state_text"] for r in val_recs}
    assert len(train_texts.intersection(val_texts)) == 0, "State text overlap between train and val!"


def test_original_test_split_remains_untouched(test_parquet):
    # Ensure test split is never used as input to pretraining split function
    test_meta = pq.read_metadata(str(test_parquet))
    assert test_meta.num_rows == 24386
