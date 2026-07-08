#!/usr/bin/env python3
"""
123_generate_lifecycle_report.py
Generate comprehensive lifecycle prediction experiment report.

Usage:
  python code/scripts/123_generate_lifecycle_report.py
  python code/scripts/123_generate_lifecycle_report.py --format markdown
"""
import os, sys, json
from pathlib import Path, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction")
REPORT_DIR = os.path.join(OUT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

MODELS = ["mlp", "lstm", "transformer", "mamba", "mamba_lstm", "lag_aware_fusion"]
HORIZONS = [1, 2, 4, 8]
SEEDS = [42, 43, 44]


def load_collected_results():
    """Load previously collected CSV results."""
    dfs = {}
    for task in ["lifecycle", "pseudotime_regression", "direction", "ablation"]:
        csv_path = os.path.join(REPORT_DIR, f"{task}_results.csv")
        if os.path.exists(csv_path):
            dfs[task] = pd.read_csv(csv_path)
    return dfs


def aggregate_metrics(df, metric_cols, group_cols):
    """Aggregate metrics with mean and std across seeds."""
    valid = df[df["status"] == "completed"].copy()
    if valid.empty:
        return pd.DataFrame()

    agg_dict = {}
    for col in metric_cols:
        if col in valid.columns:
            valid[col] = pd.to_numeric(valid[col], errors="coerce")
            agg_dict[col] = ["mean", "std"]

    if not agg_dict:
        return pd.DataFrame()

    grouped = valid.groupby(group_cols).agg(agg_dict).reset_index()
    return grouped


def generate_markdown_report(dfs):
    """Generate markdown-format report."""
    lines = []
    lines.append(f"# Lifecycle Prediction Experiment Report")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Lifecycle classification
    if "lifecycle" in dfs and not dfs["lifecycle"].empty:
        df = dfs["lifecycle"]
        lines.append("## 1. Future Lifecycle Stage Prediction")
        lines.append("")

        agg = aggregate_metrics(df, ["accuracy", "macro_f1", "balanced_accuracy"], ["model", "horizon"])
        if not agg.empty:
            lines.append("### By Model and Horizon (mean ± std across 3 seeds)")
            lines.append("")
            lines.append("| Model | Horizon | Accuracy | Macro F1 | Balanced Acc |")
            lines.append("|-------|---------|----------|----------|-------------|")
            for _, row in agg.iterrows():
                lines.append(f"| {row['model']} | {int(row['horizon'])} | "
                             f"{row['accuracy']['mean']:.4f}±{row['accuracy']['std']:.4f} | "
                             f"{row['macro_f1']['mean']:.4f}±{row['macro_f1']['std']:.4f} | "
                             f"{row['balanced_accuracy']['mean']:.4f}±{row['balanced_accuracy']['std']:.4f} |")
            lines.append("")

        # By model (average across horizons)
        agg_model = aggregate_metrics(df, ["accuracy", "macro_f1"], ["model"])
        if not agg_model.empty:
            best_model = agg_model.loc[agg_model[("accuracy", "mean")].idxmax()]
            lines.append(f"**Best model:** {best_model['model']} "
                         f"(acc={best_model[('accuracy','mean')]:.4f}±{best_model[('accuracy','std')]:.4f})")
            lines.append("")

        # Completion stats
        completed = (df["status"] == "completed").sum()
        failed = (df["status"] == "failed").sum()
        total = len(df)
        lines.append(f"**Status:** {completed}/{total} completed, {failed} failed")
        lines.append("")

    # Pseudotime regression
    if "pseudotime_regression" in dfs and not dfs["pseudotime_regression"].empty:
        df = dfs["pseudotime_regression"]
        lines.append("## 2. Future Pseudotime Regression")
        lines.append("")

        agg = aggregate_metrics(df, ["mae", "rmse", "pearson_r", "r2"], ["model", "horizon"])
        if not agg.empty:
            lines.append("| Model | Horizon | MAE | RMSE | Pearson r | R² |")
            lines.append("|-------|---------|-----|------|-----------|----|")
            for _, row in agg.iterrows():
                lines.append(f"| {row['model']} | {int(row['horizon'])} | "
                             f"{row['mae']['mean']:.4f}±{row['mae']['std']:.4f} | "
                             f"{row['rmse']['mean']:.4f}±{row['rmse']['std']:.4f} | "
                             f"{row['pearson_r']['mean']:.4f}±{row['pearson_r']['std']:.4f} | "
                             f"{row['r2']['mean']:.4f}±{row['r2']['std']:.4f} |")
            lines.append("")

    # Direction prediction
    if "direction" in dfs and not dfs["direction"].empty:
        df = dfs["direction"]
        lines.append("## 3. Trajectory Direction Prediction")
        lines.append("")

        agg = aggregate_metrics(df, ["accuracy", "macro_f1"], ["model", "horizon"])
        if not agg.empty:
            lines.append("| Model | Horizon | Accuracy | Macro F1 |")
            lines.append("|-------|---------|----------|----------|")
            for _, row in agg.iterrows():
                lines.append(f"| {row['model']} | {int(row['horizon'])} | "
                             f"{row['accuracy']['mean']:.4f}±{row['accuracy']['std']:.4f} | "
                             f"{row['macro_f1']['mean']:.4f}±{row['macro_f1']['std']:.4f} |")
            lines.append("")

    # Ablation
    if "ablation" in dfs and not dfs["ablation"].empty:
        df = dfs["ablation"]
        lines.append("## 4. Ablation Study")
        lines.append("")

        agg = aggregate_metrics(df, ["accuracy", "macro_f1"], ["mode"])
        if not agg.empty:
            agg_sorted = agg.sort_values(("accuracy", "mean"), ascending=False)
            lines.append("| Mode | Accuracy | Macro F1 |")
            lines.append("|------|----------|----------|")
            for _, row in agg_sorted.iterrows():
                lines.append(f"| {row['mode']} | "
                             f"{row['accuracy']['mean']:.4f}±{row['accuracy']['std']:.4f} | "
                             f"{row['macro_f1']['mean']:.4f}±{row['macro_f1']['std']:.4f} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="markdown", choices=["markdown", "json", "text"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Generating Lifecycle Report")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    dfs = load_collected_results()
    print(f"  Loaded {len(dfs)} result sets")

    if args.format == "markdown":
        report = generate_markdown_report(dfs)
        report_path = os.path.join(REPORT_DIR, "lifecycle_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Report saved: {report_path}")
        print(f"\n{report[:2000]}")
    elif args.format == "json":
        # Dump as structured JSON
        json_data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for name, df in dfs.items():
            if not df.empty:
                json_data[name] = df.to_dict(orient="records")
        report_path = os.path.join(REPORT_DIR, "lifecycle_report.json")
        with open(report_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"  Report saved: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
