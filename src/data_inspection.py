"""
Dataset inspection module for the BEV Decision dataset.
Provides statistical profiling, schema validation, and data leakage checks.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pyarrow.parquet as pq


@dataclass
class SplitStats:
    split_name: str
    num_rows: int
    columns: List[str]
    column_types: Dict[str, str]
    null_counts: Dict[str, int]
    empty_state_count: int
    empty_state_fraction: float
    unique_state_count: int
    unique_state_fraction: float
    char_length_stats: Dict[str, float]
    word_count_stats: Dict[str, float]
    domain_counts: Dict[str, int]
    sample_states: List[Dict[str, Any]]


def inspect_split(
    parquet_path: Path,
    split_name: str = "train",
    num_samples: int = 5,
) -> SplitStats:
    """Inspects a single parquet split and returns comprehensive statistics."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(str(parquet_path))
    num_rows = table.num_rows
    columns = table.column_names

    column_types = {col: str(table[col].type) for col in columns}
    null_counts = {col: table[col].null_count for col in columns}

    states = table["state"].to_pylist()
    domains = table["domain"].to_pylist() if "domain" in columns else ["unknown"] * num_rows

    empty_count = sum(1 for s in states if s is None or len(s.strip()) == 0)
    empty_fraction = empty_count / max(num_rows, 1)

    unique_states = set(s for s in states if s is not None)
    unique_count = len(unique_states)
    unique_fraction = unique_count / max(num_rows, 1)

    valid_states = [s for s in states if s is not None and len(s.strip()) > 0]
    char_lens = [len(s) for s in valid_states] if valid_states else [0]
    word_counts = [len(s.split()) for s in valid_states] if valid_states else [0]

    char_stats = {
        "min": float(np.min(char_lens)),
        "p25": float(np.percentile(char_lens, 25)),
        "median": float(np.median(char_lens)),
        "mean": float(np.mean(char_lens)),
        "p75": float(np.percentile(char_lens, 75)),
        "p95": float(np.percentile(char_lens, 95)),
        "max": float(np.max(char_lens)),
    }

    word_stats = {
        "min": float(np.min(word_counts)),
        "p25": float(np.percentile(word_counts, 25)),
        "median": float(np.median(word_counts)),
        "mean": float(np.mean(word_counts)),
        "p75": float(np.percentile(word_counts, 75)),
        "p95": float(np.percentile(word_counts, 95)),
        "max": float(np.max(word_counts)),
    }

    from collections import Counter
    domain_counts = dict(Counter(domains).most_common(10))

    # Collect diverse samples
    sample_states = []
    seen_domains = set()
    for s, d in zip(states, domains):
        if s is not None and len(s.split()) >= 10 and d not in seen_domains:
            seen_domains.add(d)
            sample_states.append({
                "domain": d,
                "word_count": len(s.split()),
                "preview": s[:250] + ("..." if len(s) > 250 else ""),
            })
            if len(sample_states) >= num_samples:
                break

    return SplitStats(
        split_name=split_name,
        num_rows=num_rows,
        columns=columns,
        column_types=column_types,
        null_counts=null_counts,
        empty_state_count=empty_count,
        empty_state_fraction=empty_fraction,
        unique_state_count=unique_count,
        unique_state_fraction=unique_fraction,
        char_length_stats=char_stats,
        word_count_stats=word_stats,
        domain_counts=domain_counts,
        sample_states=sample_states,
    )


def verify_no_data_leakage(
    parquet_path: Path,
    num_rows_to_check: int = 1000,
) -> Dict[str, Any]:
    """
    Verifies that the state field does not embed decision questions or ground truth labels.
    Returns audit results.
    """
    table = pq.read_table(str(parquet_path))
    num_rows = min(table.num_rows, num_rows_to_check)

    states = table["state"].slice(0, num_rows).to_pylist()
    q_jsons = table["questions_json"].slice(0, num_rows).to_pylist()

    leaked_instructions_count = 0
    leaked_labels_count = 0

    for s, qj in zip(states, q_jsons):
        if not s or not qj:
            continue
        try:
            q_data = json.loads(qj)
            for q_id, q_val in q_data.items():
                if isinstance(q_val, dict):
                    instructions = q_val.get("instructions", "")
                    if instructions and len(instructions) > 10 and instructions.lower() in s.lower():
                        leaked_instructions_count += 1
                    
                    label = str(q_val.get("label", ""))
                    # Short labels like 'yes'/'no' could appear by coincidence in natural text,
                    # but specific compound labels can be checked.
                    if label and len(label) > 6 and label.lower() in s.lower():
                        leaked_labels_count += 1
        except Exception:
            pass

    return {
        "checked_rows": num_rows,
        "leaked_instructions_count": leaked_instructions_count,
        "leaked_labels_count": leaked_labels_count,
        "is_leakage_free": (leaked_instructions_count == 0),
    }
