#!/usr/bin/env python3
"""
CLI script to inspect the BEV Decision dataset for Phase 1 JEPA pretraining.
"""

import argparse
from pathlib import Path
import json
import sys

# Ensure src is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_inspection import inspect_split, verify_no_data_leakage


def main():
    parser = argparse.ArgumentParser(description="Inspect BEV Decision dataset")
    parser.add_argument("--data-dir", type=str, default="data/bev-decision", help="Path to dataset directory")
    parser.add_argument("--output-json", type=str, default="outputs/phase1/metrics/dataset_inspection.json", help="Path to output JSON")
    args = parser.parse_args()

    data_dir = project_root / args.data_dir
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"

    print("=" * 60)
    print("      BEV DECISION DATASET INSPECTION REPORT (PHASE 1)      ")
    print("=" * 60)

    train_stats = inspect_split(train_path, split_name="train")
    test_stats = inspect_split(test_path, split_name="test")

    for stats in [train_stats, test_stats]:
        print(f"\n--- Split: {stats.split_name.upper()} ---")
        print(f"Total Rows: {stats.num_rows:,}")
        print(f"Columns: {', '.join(stats.columns)}")
        print("Data Types:")
        for col, col_t in stats.column_types.items():
            print(f"  {col}: {col_t} (nulls: {stats.null_counts[col]})")
        print(f"Unique States: {stats.unique_state_count:,} ({stats.unique_state_fraction * 100:.2f}%)")
        print(f"Empty States: {stats.empty_state_count} ({stats.empty_state_fraction * 100:.3f}%)")
        print("Word Count Distribution:")
        print(f"  min={stats.word_count_stats['min']:.0f}, p25={stats.word_count_stats['p25']:.0f}, "
              f"median={stats.word_count_stats['median']:.0f}, mean={stats.word_count_stats['mean']:.1f}, "
              f"p75={stats.word_count_stats['p75']:.0f}, p95={stats.word_count_stats['p95']:.0f}, "
              f"max={stats.word_count_stats['max']:.0f}")
        print("Top Domains:")
        for dom, cnt in stats.domain_counts.items():
            print(f"  - {dom}: {cnt:,} ({cnt / stats.num_rows * 100:.1f}%)")
        print("\nRepresentative State Samples:")
        for idx, s in enumerate(stats.sample_states, 1):
            print(f"  [{idx}] Domain: {s['domain']} | Words: {s['word_count']}")
            print(f"      Preview: {s['preview']}")

    print("\n" + "=" * 60)
    print("DATA LEAKAGE AUDIT (Checking state against questions_json)")
    print("=" * 60)
    audit = verify_no_data_leakage(train_path, num_rows_to_check=2000)
    print(f"Audited rows: {audit['checked_rows']}")
    print(f"Question instruction leakage count: {audit['leaked_instructions_count']}")
    print(f"Candidate label leakage count: {audit['leaked_labels_count']}")
    print(f"Status: {'PASSED (Zero question leakage into state)' if audit['is_leakage_free'] else 'WARNING'}")

    out_path = project_root / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "train": train_stats.__dict__,
        "test": test_stats.__dict__,
        "leakage_audit": audit,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nInspection JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
