"""
Data processing pipeline for Phase 1 JEPA pretraining.
Extracts label-free state text, deduplicates/groups by state, and creates
deterministic train/validation splits without data leakage.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import random
import pyarrow.parquet as pq


def hash_state(text: str) -> str:
    """Generates a consistent MD5 hash for state text deduplication/grouping."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def load_and_clean_states(
    parquet_path: Path,
    min_word_count: int = 6,
) -> List[Dict[str, Any]]:
    """
    Loads rows from parquet, isolates the state text and domain, and filters out
    empty or excessively short states.

    CRITICAL LEAKAGE RULE:
    Explicitly drops questions_json, question_types, question_count, and any decision labels.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(str(parquet_path))
    states = table["state"].to_pylist()
    domains = table["domain"].to_pylist() if "domain" in table.column_names else ["unknown"] * len(states)

    cleaned_records: List[Dict[str, Any]] = []
    seen_hashes = set()

    for idx, (raw_state, domain) in enumerate(zip(states, domains)):
        if raw_state is None:
            continue
        state_str = str(raw_state).strip()
        words = state_str.split()
        if len(words) < min_word_count:
            continue

        s_hash = hash_state(state_str)
        cleaned_records.append({
            "state_id": f"row_{idx}_{s_hash[:8]}",
            "state_hash": s_hash,
            "state_text": state_str,
            "domain": domain,
            "word_count": len(words),
        })

    return cleaned_records


def split_states(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.10,
    seed: int = 42,
    deduplicate_identical_states: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Splits records into internal train and validation partitions at the state-group level.
    Ensures that identical states never appear in both train and validation splits.

    Returns:
        train_records: List of state dictionaries
        val_records: List of state dictionaries
        split_metadata: Dictionary with split counts and statistics
    """
    rng = random.Random(seed)

    # Group by state_hash
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        h = r["state_hash"]
        if h not in groups:
            groups[h] = []
        groups[h].append(r)

    unique_hashes = sorted(list(groups.keys()))
    rng.shuffle(unique_hashes)

    num_val_groups = int(len(unique_hashes) * val_ratio)
    val_hashes = set(unique_hashes[:num_val_groups])
    train_hashes = set(unique_hashes[num_val_groups:])

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    for h, recs in groups.items():
        chosen = [recs[0]] if deduplicate_identical_states else recs
        if h in val_hashes:
            val_records.extend(chosen)
        else:
            train_records.extend(chosen)

    # Verification: check set intersection of state texts
    train_texts = {r["state_text"] for r in train_records}
    val_texts = {r["state_text"] for r in val_records}
    intersection = train_texts.intersection(val_texts)
    assert len(intersection) == 0, f"Data leakage: {len(intersection)} states overlap between train and val!"

    split_metadata = {
        "total_input_records": len(records),
        "total_unique_groups": len(unique_hashes),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "val_ratio": val_ratio,
        "seed": seed,
        "deduplicated": deduplicate_identical_states,
        "overlap_count": len(intersection),
    }

    return train_records, val_records, split_metadata


def verify_record_schema(record: Dict[str, Any]) -> bool:
    """Verifies that record contains ONLY permitted state fields and zero question/label fields."""
    forbidden_keys = {
        "questions_json", "question_types", "question_count",
        "label", "choice", "score", "noul", "target_label", "answer"
    }
    keys = set(record.keys())
    if forbidden_keys.intersection(keys):
        return False
    return "state_text" in record
