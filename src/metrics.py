"""
Metrics tracking, history logging, and visualization plotting for JEPA pretraining.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import csv
import matplotlib.pyplot as plt


class MetricsTracker:
    """Tracks training and validation metrics, handles history, and creates plots."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_dir = output_dir / "metrics"
        self.plots_dir = output_dir / "plots"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_history: List[Dict[str, Any]] = []
        self.step_history: List[Dict[str, Any]] = []

    def record_step(self, step_data: Dict[str, Any]) -> None:
        self.step_history.append(step_data)

    def record_epoch(self, epoch_data: Dict[str, Any]) -> None:
        self.epoch_history.append(epoch_data)

    def save_metrics(self) -> Tuple[Path, Path]:
        """Saves epoch history to JSON and CSV."""
        json_path = self.metrics_dir / "phase1_metrics.json"
        with open(json_path, "w") as f:
            json.dump({
                "epoch_history": self.epoch_history,
                "total_epochs": len(self.epoch_history),
            }, f, indent=2)

        csv_path = self.metrics_dir / "phase1_metrics.csv"
        if self.epoch_history:
            keys = list(self.epoch_history[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.epoch_history)

        return json_path, csv_path

    def generate_plots(self) -> List[Path]:
        """
        Generates required diagnostic plots:
        1. Train & Val Loss vs Epoch
        2. Mean Latent Dimension Std vs Epoch
        3. Mean Latent Vector Norm vs Epoch
        4. Pairwise Cosine Similarity vs Epoch
        """
        if not self.epoch_history:
            return []

        epochs = [d["epoch"] for d in self.epoch_history]
        generated_plots = []

        # 1. Loss vs Epoch
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, [d["train_loss"] for d in self.epoch_history], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
        plt.plot(epochs, [d["val_loss"] for d in self.epoch_history], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
        plt.title("JEPA Latent Prediction Loss vs. Epoch", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Loss (Normalized MSE)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        loss_plot = self.plots_dir / "loss_vs_epoch.png"
        plt.savefig(loss_plot, dpi=200)
        plt.close()
        generated_plots.append(loss_plot)

        # 2. Mean Latent Std vs Epoch (Representation Collapse Diagnostic)
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, [d["val_mean_latent_std"] for d in self.epoch_history], "d-", label="Mean Latent Std (Val)", color="#2ca02c", linewidth=2)
        plt.axhline(y=0.05, color="red", linestyle=":", label="Collapse Alert Threshold (<0.05)")
        plt.title("Mean Latent Standard Deviation vs. Epoch", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Mean Latent Dimension Std", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        std_plot = self.plots_dir / "latent_std_vs_epoch.png"
        plt.savefig(std_plot, dpi=200)
        plt.close()
        generated_plots.append(std_plot)

        # 3. Mean Latent Vector Norm vs Epoch
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, [d["val_mean_latent_norm"] for d in self.epoch_history], "^-", label="Mean Vector Norm", color="#9467bd", linewidth=2)
        plt.title("Mean Latent Vector Norm vs. Epoch", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("L2 Norm", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        norm_plot = self.plots_dir / "latent_norm_vs_epoch.png"
        plt.savefig(norm_plot, dpi=200)
        plt.close()
        generated_plots.append(norm_plot)

        # 4. Pairwise Cosine Similarity vs Epoch
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, [d["val_mean_pairwise_similarity"] for d in self.epoch_history], "x-", label="Pairwise Cosine Sim", color="#d62728", linewidth=2)
        plt.axhline(y=0.95, color="black", linestyle=":", label="Collapse Sim Alert (>0.95)")
        plt.title("Mean Pairwise Cosine Similarity vs. Epoch", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Cosine Similarity", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        sim_plot = self.plots_dir / "pairwise_similarity_vs_epoch.png"
        plt.savefig(sim_plot, dpi=200)
        plt.close()
        generated_plots.append(sim_plot)

        return generated_plots
