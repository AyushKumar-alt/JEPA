"""
Automated Orchestration Suite for Downstream Label-Efficiency Experiments.
Executes matched fine-tuning for Model A (Supervised), Model B (JEPA), and Model C (MLM)
across label fractions (10%, 25%, 50%, 100%) and evaluates each on the official test split.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.train_downstream import train_downstream_model
from src.evaluate_downstream import evaluate_test_split


def run_experiment_suite(
    models: List[str] = ["supervised", "jepa", "mlm"],
    fractions: List[float] = [0.10, 0.25, 0.50, 1.00],
    seeds: List[int] = [42],
    epochs: int = 3,
    patience: int = 2,
    batch_size: int = 64,
    lr: float = 3e-5,
    max_train_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("STARTING COMPREHENSIVE LABEL-EFFICIENCY EXPERIMENT SUITE")
    print(f"Models: {models}")
    print(f"Fractions: {[f'{int(f*100)}%' for f in fractions]}")
    print(f"Seeds: {seeds}")
    print(f"Epochs per run: {epochs} (Early Stopping Patience: {patience})")
    print("=" * 70)

    test_parquet = project_root / "data" / "bev-decision" / "test.parquet"
    assert test_parquet.exists(), f"Missing official test split at {test_parquet}"

    all_results: List[Dict[str, Any]] = []
    output_dir = project_root / "outputs" / "phase2"
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    suite_start = time.time()
    total_runs = len(models) * len(fractions) * len(seeds)
    run_idx = 0

    for seed in seeds:
        for frac in fractions:
            for model_type in models:
                run_idx += 1
                frac_pct = int(frac * 100)
                print("\n" + "#" * 60)
                print(f"RUN [{run_idx}/{total_runs}]: Model={model_type.upper()} | Fraction={frac_pct}% | Seed={seed}")
                print("#" * 60)

                # Reset peak memory stats
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                run_start = time.time()
                # 1. Train downstream model
                train_res = train_downstream_model(
                    model_type=model_type,
                    label_fraction=frac,
                    seed=seed,
                    project_root=project_root,
                    epochs=epochs,
                    patience=patience,
                    batch_size=batch_size,
                    lr=lr,
                    max_train_rows=max_train_rows,
                )

                peak_vram_gb = 0.0
                if torch.cuda.is_available():
                    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

                best_ckpt = Path(train_res["checkpoint_path"])
                assert best_ckpt.exists(), f"Checkpoint not found at {best_ckpt}"

                # 2. Evaluate on untouched official test split
                test_res = evaluate_test_split(
                    checkpoint_path=best_ckpt,
                    test_parquet=test_parquet,
                    batch_size=batch_size,
                )

                # Save individual test json
                test_json_path = metrics_dir / f"{model_type}_frac{frac_pct}_seed{seed}_test_metrics.json"
                with open(test_json_path, "w") as f:
                    json.dump(test_res, f, indent=2)

                combined_record = {
                    "model_type": model_type,
                    "label_fraction": frac,
                    "label_fraction_pct": frac_pct,
                    "seed": seed,
                    "train_duration_sec": train_res["duration_sec"],
                    "peak_vram_gb": round(peak_vram_gb, 2),
                    "best_epoch": train_res["best_epoch"],
                    "best_val_loss": round(train_res["best_val_loss"], 4),
                    "best_val_acc": round(train_res["best_val_acc"], 4),
                    "test_overall_acc": round(test_res["overall_accuracy"], 4),
                    "test_overall_macro_f1": round(test_res["overall_macro_f1"], 4),
                    "test_choice_acc": round(test_res["task_accuracies"].get("choice", 0.0), 4),
                    "test_choice_macro_f1": round(test_res["task_macro_f1s"].get("choice", 0.0), 4),
                    "test_noul_acc": round(test_res["task_accuracies"].get("noul", 0.0), 4),
                    "test_noul_macro_f1": round(test_res["task_macro_f1s"].get("noul", 0.0), 4),
                    "test_score_acc": round(test_res["task_accuracies"].get("score", 0.0), 4),
                    "test_score_macro_f1": round(test_res["task_macro_f1s"].get("score", 0.0), 4),
                    "test_score_mae": round(test_res["score_mae"], 4),
                    "eval_duration_sec": round(test_res["eval_duration_sec"], 2),
                    "checkpoint_name": best_ckpt.name,
                }
                all_results.append(combined_record)

                # Incremental summary save
                summary_json = metrics_dir / "label_efficiency_summary.json"
                with open(summary_json, "w") as f:
                    json.dump(all_results, f, indent=2)

    total_suite_time = time.time() - suite_start
    print("\n" + "=" * 70)
    print(f"EXPERIMENT SUITE FINISHED IN {total_suite_time/60:.1f} MINUTES")
    print("=" * 70)

    # Write CSV summary
    summary_csv = metrics_dir / "label_efficiency_summary.csv"
    if all_results:
        keys = list(all_results[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Saved CSV summary to {summary_csv}")

    # Generate Plots
    generate_summary_plots(all_results, plots_dir)

    return all_results


def generate_summary_plots(results: List[Dict[str, Any]], plots_dir: Path):
    """Generates comparison curves for test accuracy and macro-F1 across label fractions."""
    if not results:
        return

    fractions = sorted(list(set(r["label_fraction_pct"] for r in results)))
    models = sorted(list(set(r["model_type"] for r in results)))

    colors = {"supervised": "#3b82f6", "jepa": "#10b981", "mlm": "#f59e0b"}
    labels = {
        "supervised": "Model A: Supervised Baseline",
        "jepa": "Model B: JEPA-Pretrained",
        "mlm": "Model C: MLM-Pretrained",
    }

    # Plot 1: Test Accuracy vs Label Fraction
    plt.figure(figsize=(9, 5.5), dpi=300)
    for m in ["supervised", "jepa", "mlm"]:
        m_runs = [r for r in results if r["model_type"] == m]
        if not m_runs:
            continue
        xs = []
        ys = []
        for frac in fractions:
            vals = [r["test_overall_acc"] * 100 for r in m_runs if r["label_fraction_pct"] == frac]
            if vals:
                xs.append(frac)
                ys.append(float(np.mean(vals)))
        plt.plot(xs, ys, marker="o", linewidth=2.5, markersize=8, label=labels.get(m, m), color=colors.get(m, "gray"))

    plt.title("Label-Efficiency Comparison on Official Held-Out Test Split", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Label Fraction (%)", fontsize=11, fontweight="medium")
    plt.ylabel("Overall Test Accuracy (%)", fontsize=11, fontweight="medium")
    plt.xticks(fractions, [f"{f}%" for f in fractions])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "label_efficiency_accuracy.png")
    plt.close()

    # Plot 2: Test Macro-F1 vs Label Fraction
    plt.figure(figsize=(9, 5.5), dpi=300)
    for m in ["supervised", "jepa", "mlm"]:
        m_runs = [r for r in results if r["model_type"] == m]
        if not m_runs:
            continue
        xs = []
        ys = []
        for frac in fractions:
            vals = [r["test_overall_macro_f1"] * 100 for r in m_runs if r["label_fraction_pct"] == frac]
            if vals:
                xs.append(frac)
                ys.append(float(np.mean(vals)))
        plt.plot(xs, ys, marker="s", linewidth=2.5, markersize=8, label=labels.get(m, m), color=colors.get(m, "gray"))

    plt.title("Label-Efficiency Macro-F1 on Official Held-Out Test Split", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Label Fraction (%)", fontsize=11, fontweight="medium")
    plt.ylabel("Overall Test Macro-F1 (%)", fontsize=11, fontweight="medium")
    plt.xticks(fractions, [f"{f}%" for f in fractions])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "label_efficiency_macro_f1.png")
    plt.close()

    # Plot 3: Task Breakdown at 10% and 100%
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)
    for idx, target_frac in enumerate([10, 100]):
        ax = axes[idx]
        sub = [r for r in results if r["label_fraction_pct"] == target_frac]
        if not sub:
            continue
        tasks = ["choice", "noul", "score"]
        x = np.arange(len(tasks))
        width = 0.25

        for i, m in enumerate(["supervised", "jepa", "mlm"]):
            m_data = [r for r in sub if r["model_type"] == m]
            if not m_data:
                continue
            rec = m_data[0]
            accs = [rec["test_choice_acc"] * 100, rec["test_noul_acc"] * 100, rec["test_score_acc"] * 100]
            ax.bar(x + (i - 1) * width, accs, width, label=labels.get(m, m), color=colors.get(m, "gray"), alpha=0.85)

        ax.set_title(f"Per-Task Test Accuracy @ {target_frac}% Labels", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["CHOICE", "NOUL", "SCORE"], fontweight="medium")
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        if idx == 0:
            ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir / "task_performance_breakdown.png")
    plt.close()
    print(f"Generated comparison plots in {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--models", nargs="+", type=str, default=["supervised", "jepa", "mlm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    run_experiment_suite(
        models=args.models,
        fractions=args.fractions,
        seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_rows=args.max_rows,
    )
