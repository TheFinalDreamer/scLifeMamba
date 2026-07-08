#!/usr/bin/env python3
"""
124_generate_lifecycle_figures.py
Generate publication-quality figures for lifecycle prediction results.

Usage:
  python code/scripts/124_generate_lifecycle_figures.py
  python code/scripts/124_generate_lifecycle_figures.py --fig accuracy_bar
"""
import os, sys, json
from pathlib import Path, argparse
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction")
REPORT_DIR = os.path.join(OUT_DIR, "reports")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_results():
    """Load all collected CSV results."""
    dfs = {}
    for task in ["lifecycle", "pseudotime_regression", "direction", "ablation"]:
        csv_path = os.path.join(REPORT_DIR, f"{task}_results.csv")
        if os.path.exists(csv_path):
            dfs[task] = pd.read_csv(csv_path)
    return dfs


def plot_accuracy_bars(df, save_path):
    """Bar chart: accuracy by model and horizon."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        completed = df[df["status"] == "completed"].copy()
        if completed.empty:
            print("  No completed experiments to plot")
            return

        completed["accuracy"] = pd.to_numeric(completed["accuracy"], errors="coerce")
        completed["label"] = completed["model"] + " h=" + completed["horizon"].astype(str)

        fig, ax = plt.subplots(figsize=(14, 6))
        models_order = ["mlp", "lstm", "transformer", "mamba", "mamba_lstm", "lag_aware_fusion"]
        completed["model"] = pd.Categorical(completed["model"], categories=models_order, ordered=True)
        completed = completed.sort_values(["horizon", "model"])

        sns.barplot(data=completed, x="label", y="accuracy", hue="model", ax=ax, dodge=False)
        ax.set_title("Lifecycle Stage Prediction Accuracy by Model and Horizon")
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.5, label="Random baseline (25%)")
        ax.legend(title="Model")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except ImportError:
        print("  matplotlib/seaborn not available, skipping plot")


def plot_horizon_degradation(df, save_path):
    """Line plot: accuracy degradation with increasing horizon."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        completed = df[df["status"] == "completed"].copy()
        if completed.empty:
            return

        completed["accuracy"] = pd.to_numeric(completed["accuracy"], errors="coerce")

        fig, ax = plt.subplots(figsize=(10, 6))
        for model in completed["model"].unique():
            model_data = completed[completed["model"] == model]
            means = model_data.groupby("horizon")["accuracy"].mean()
            stds = model_data.groupby("horizon")["accuracy"].std()
            ax.errorbar(means.index, means.values, yerr=stds.values, label=model, marker="o", capsize=3)

        ax.set_title("Accuracy Degradation with Prediction Horizon")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot")


def plot_ablation_comparison(df, save_path):
    """Bar chart: ablation mode comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        completed = df[df["status"] == "completed"].copy()
        if completed.empty:
            return

        completed["accuracy"] = pd.to_numeric(completed["accuracy"], errors="coerce")
        mode_means = completed.groupby("mode")["accuracy"].agg(["mean", "std"]).reset_index()
        mode_means = mode_means.sort_values("mean", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(mode_means)), mode_means["mean"], yerr=mode_means["std"], capsize=3)
        ax.set_xticks(range(len(mode_means)))
        ax.set_xticklabels(mode_means["mode"], rotation=45, ha="right")
        ax.set_title("Ablation Study: Fusion Mode vs. Accuracy")
        ax.set_ylabel("Accuracy")
        ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.5, label="Random baseline")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except ImportError:
        print("  matplotlib/seaborn not available, skipping plot")


def plot_confusion_matrix_aggregate(df, save_path):
    """Plot aggregate confusion matrix across best model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        completed = df[df["status"] == "completed"].copy()
        if completed.empty:
            return

        # Find best model at horizon=4
        h4 = completed[completed["horizon"] == 4]
        if h4.empty:
            return

        best = h4.loc[h4["accuracy"].idxmax()]

        # Load the actual confusion matrix
        metrics_path = os.path.join(
            OUT_DIR, "lifecycle_prediction", best["model"],
            f"h{int(best['horizon'])}_s{int(best['seed'])}_ctx8", "metrics.json"
        )
        if not os.path.exists(metrics_path):
            return

        with open(metrics_path) as f:
            metrics = json.load(f)
        cm = np.array(metrics.get("confusion_matrix", []))

        if cm.size == 0:
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        n_classes = cm.shape[0]
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        stage_names = ["Early", "Transition", "Late", "Terminal"][:n_classes]
        ax.set_xticklabels(stage_names)
        ax.set_yticklabels(stage_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_title(f"Confusion Matrix: {best['model']} h={int(best['horizon'])}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except (ImportError, Exception) as e:
        print(f"  Could not plot confusion matrix: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", default="all",
                        choices=["all", "accuracy_bar", "horizon_degradation", "ablation", "confusion"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Generating Lifecycle Figures")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    dfs = load_results()
    print(f"  Loaded {len(dfs)} result sets")

    if "lifecycle" in dfs:
        df = dfs["lifecycle"]
        if args.fig in ("all", "accuracy_bar"):
            plot_accuracy_bars(df, os.path.join(FIG_DIR, "fig1_accuracy_bars.png"))
        if args.fig in ("all", "horizon_degradation"):
            plot_horizon_degradation(df, os.path.join(FIG_DIR, "fig2_horizon_degradation.png"))
        if args.fig in ("all", "confusion"):
            plot_confusion_matrix_aggregate(df, os.path.join(FIG_DIR, "fig3_confusion_matrix.png"))

    if "ablation" in dfs and args.fig in ("all", "ablation"):
        plot_ablation_comparison(dfs["ablation"], os.path.join(FIG_DIR, "fig4_ablation_comparison.png"))

    print(f"\n  Figures saved to: {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
