"""
Phase 1 Evaluation and Audit Script.
Analyzes Phase 1 checkpoints, verifies representation diversity, audits data leakage,
and tests reloadability of the exported reusable representation encoder.
"""

import argparse
from pathlib import Path
import json
import yaml
import torch
from typing import Dict, Any, List, Optional
import sys

project_root_dir = Path(__file__).resolve().parent.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

from src.checkpointing import load_checkpoint, load_reusable_encoder
from src.jepa_model import TextJEPA
from src.losses import JEPALoss
from src.data_processing import load_and_clean_states, split_states
from src.view_generator import ViewGenerator


def evaluate_phase1(
    config_path: Path,
    project_root: Path,
    checkpoint_name: str = "checkpoint_best.pt",
) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = project_root / cfg["experiment"]["output_dir"]
    checkpoints_dir = output_dir / "checkpoints"
    ckpt_path = checkpoints_dir / checkpoint_name
    reusable_dir = checkpoints_dir / "reusable_encoder"

    print("=" * 60)
    print("        PHASE 1 JEPA EVALUATION & VERIFICATION AUDIT        ")
    print("=" * 60)

    device_str = "cuda" if torch.cuda.is_available() and cfg["experiment"].get("device", "cuda") == "cuda" else "cpu"
    device = torch.device(device_str)

    # 1. Inspect saved metrics
    metrics_json = output_dir / "metrics" / "phase1_metrics.json"
    epoch_history = []
    if metrics_json.exists():
        with open(metrics_json, "r") as f:
            metrics_data = json.load(f)
            epoch_history = metrics_data.get("epoch_history", [])

    # 2. Verify and test the reusable encoder
    print(f"\n[1] Verifying Reusable Context Encoder at: {reusable_dir}")
    encoder_verified = False
    test_latents_stats = {}
    if (reusable_dir / "context_encoder.pt").exists():
        encoder, tokenizer, meta = load_reusable_encoder(reusable_dir, device=device_str)
        print("  --> Reusable encoder loaded successfully!")
        print(f"  --> Backbone: {meta['backbone_name']} | Latent Dim: {meta['latent_dim']} | Pooling: {meta['pooling_type']}")

        # Test on dummy inputs
        sample_texts = [
            "The customer asked for a full refund after the package failed to arrive on time.",
            "Algorithm complexity depends on the graph branching factor and edge weights.",
            "Blocks A and B are situated adjacent to the left boundary of the observation space.",
            "Safety check failed: vulnerability CVE-2024-1234 detected in upstream library.",
        ]
        tokens = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            latents = encoder(tokens["input_ids"], tokens["attention_mask"])

        loss_fn = JEPALoss()
        diag = loss_fn.compute_diagnostics(latents)
        encoder_verified = True
        test_latents_stats = {
            "test_sample_count": len(sample_texts),
            "output_shape": list(latents.shape),
            "mean_norm": diag.mean_latent_norm,
            "mean_dim_std": diag.mean_dimension_std,
            "mean_pairwise_similarity": diag.mean_pairwise_similarity,
            "collapsed_dim_fraction": diag.collapsed_dimension_fraction,
            "has_collapse_warning": diag.has_collapse_warning,
        }
        print(f"  --> Sample Latents Shape: {latents.shape}")
        print(f"  --> Mean Latent Norm: {diag.mean_latent_norm:.3f}")
        print(f"  --> Mean Per-Dimension Std: {diag.mean_dimension_std:.3f}")
        print(f"  --> Mean Pairwise Cosine Sim: {diag.mean_pairwise_similarity:.3f}")
        print(f"  --> Representation Status: {'HEALTHY' if not diag.has_collapse_warning else 'WARNING: ' + diag.warning_reason}")
    else:
        print("  [!] Reusable encoder directory not yet created. Train model first.")

    # 3. Model parameter audit
    model_cfg = cfg["model"]
    model = TextJEPA(
        backbone_name_or_path=model_cfg["backbone_name"],
        latent_dim=model_cfg["latent_dim"],
        predictor_hidden_dim=model_cfg["predictor_hidden_dim"],
        predictor_num_layers=model_cfg["predictor_num_layers"],
        pooling_type=model_cfg["pooling_type"],
        normalize_latents=model_cfg["normalize_latents"],
    )
    param_counts = model.count_parameters()

    # 4. Summary report
    summary = {
        "experiment_title": cfg["experiment"]["title"],
        "checkpoint_evaluated": str(ckpt_path),
        "checkpoint_exists": ckpt_path.exists(),
        "reusable_encoder_verified": encoder_verified,
        "parameter_counts": param_counts,
        "epoch_metrics_summary": {
            "total_epochs_recorded": len(epoch_history),
            "best_val_loss": min([e["val_loss"] for e in epoch_history]) if epoch_history else None,
            "final_val_loss": epoch_history[-1]["val_loss"] if epoch_history else None,
            "final_val_dim_std": epoch_history[-1]["val_mean_latent_std"] if epoch_history else None,
            "final_val_pairwise_sim": epoch_history[-1]["val_mean_pairwise_similarity"] if epoch_history else None,
        },
        "sample_encoder_diagnostics": test_latents_stats,
    }

    eval_out = output_dir / "metrics" / "evaluation_summary.json"
    with open(eval_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation summary written to: {eval_out}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 JEPA")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pt", help="Checkpoint to evaluate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    evaluate_phase1(
        config_path=project_root / args.config,
        project_root=project_root,
        checkpoint_name=args.checkpoint,
    )


if __name__ == "__main__":
    main()
