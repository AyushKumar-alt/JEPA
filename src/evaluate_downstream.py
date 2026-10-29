"""
Comprehensive Downstream Decision Evaluation Script on Untouched Official Test Split.
Computes Accuracy, Macro-F1, per-task metrics (CHOICE, NOUL, SCORE),
Ordinal MAE for SCORE, and per-domain accuracy breakdowns.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import defaultdict

import sys
project_root_dir = Path(__file__).resolve().parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

from src.downstream_dataset import parse_downstream_dataset, DownstreamExample
from src.downstream_model import DecisionPredictionModel
from src.train_downstream import DecisionDataset, DecisionCollator


def evaluate_test_split(
    checkpoint_path: Path,
    test_parquet: Path,
    device_str: str = "cuda",
    batch_size: int = 32,
    max_test_samples: Optional[int] = None,
) -> Dict[str, Any]:
    print("=" * 60)
    print(f"OFFICIAL TEST SPLIT EVALUATION: {checkpoint_path.name}")
    print("=" * 60)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_type = ckpt.get("model_type", "supervised")

    model = DecisionPredictionModel(model_type="supervised")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    test_examples = parse_downstream_dataset(test_parquet)
    if max_test_samples and max_test_samples < len(test_examples):
        test_examples = test_examples[:max_test_samples]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collator = DecisionCollator(tokenizer, max_seq_length=256)
    loader = DataLoader(
        DecisionDataset(test_examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    correct_overall = 0
    total_overall = 0

    per_task_correct = defaultdict(int)
    per_task_total = defaultdict(int)
    per_domain_correct = defaultdict(int)
    per_domain_total = defaultdict(int)

    import time
    from sklearn.metrics import f1_score

    start_eval_time = time.time()
    all_preds = []
    all_targets = []
    all_task_types = []
    all_domains = []

    per_task_preds = defaultdict(list)
    per_task_targets = defaultdict(list)
    per_domain_preds = defaultdict(list)
    per_domain_targets = defaultdict(list)

    score_errors = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            task_types = batch["task_types"]
            domains = batch["domains"]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(input_ids, attention_mask, task_types, targets=targets)

            preds = out["predictions"]
            tgt_list = targets.cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(tgt_list)
            all_task_types.extend(task_types)
            all_domains.extend(domains)

            for p, t, tt, dom in zip(preds, tgt_list, task_types, domains):
                is_corr = (p == t)
                total_overall += 1
                if is_corr:
                    correct_overall += 1

                per_task_total[tt] += 1
                if is_corr:
                    per_task_correct[tt] += 1
                per_task_preds[tt].append(p)
                per_task_targets[tt].append(t)

                per_domain_total[dom] += 1
                if is_corr:
                    per_domain_correct[dom] += 1
                per_domain_preds[dom].append(p)
                per_domain_targets[dom].append(t)

                if tt == "score":
                    score_errors.append(abs(p - t))

    eval_duration = time.time() - start_eval_time
    overall_acc = correct_overall / max(total_overall, 1)

    task_accuracies = {
        tt: per_task_correct[tt] / max(per_task_total[tt], 1)
        for tt in per_task_total
    }
    task_macro_f1s = {
        tt: float(f1_score(per_task_targets[tt], per_task_preds[tt], average="macro", zero_division=0))
        for tt in per_task_total
    }
    score_mae = float(np.mean(score_errors)) if score_errors else 0.0

    domain_breakdown = {
        dom: {
            "accuracy": per_domain_correct[dom] / max(per_domain_total[dom], 1),
            "macro_f1": float(f1_score(per_domain_targets[dom], per_domain_preds[dom], average="macro", zero_division=0)),
            "sample_count": per_domain_total[dom],
        }
        for dom in sorted(per_domain_total.keys())
    }

    # Macro-F1 across all tasks (mean of per-task macro-F1s for balanced multi-task weighting)
    overall_macro_f1 = float(np.mean(list(task_macro_f1s.values()))) if task_macro_f1s else 0.0

    results = {
        "checkpoint": checkpoint_path.name,
        "model_type": model_type,
        "label_fraction": ckpt.get("label_fraction", 1.0),
        "seed": ckpt.get("seed", 42),
        "total_test_evaluated": total_overall,
        "overall_accuracy": overall_acc,
        "overall_macro_f1": overall_macro_f1,
        "task_accuracies": task_accuracies,
        "task_macro_f1s": task_macro_f1s,
        "score_mae": score_mae,
        "domain_breakdown": domain_breakdown,
        "eval_duration_sec": eval_duration,
        "throughput_samples_per_sec": total_overall / max(eval_duration, 1e-4),
    }

    print(f"Overall Test Accuracy: {overall_acc * 100:.2f}% | Overall Macro-F1: {overall_macro_f1 * 100:.2f}%")
    for tt, acc in task_accuracies.items():
        f1 = task_macro_f1s[tt]
        print(f"  - {tt.upper()}: Acc={acc * 100:.2f}%, Macro-F1={f1 * 100:.2f}% (Count: {per_task_total[tt]:,})")
    print(f"  - SCORE MAE: {score_mae:.3f}")
    print(f"  - Evaluation Throughput: {results['throughput_samples_per_sec']:.1f} samples/s ({eval_duration:.2f}s)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-parquet", type=str, default="data/bev-decision/test.parquet")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    test_p = Path(args.test_parquet)
    if not test_p.is_absolute():
        test_p = project_root / test_p

    results = evaluate_test_split(ckpt_path, test_p)

    out_file = args.output_json
    if out_file:
        out_p = project_root / out_file
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation results to {out_p}")
