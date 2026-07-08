#!/usr/bin/env python3
"""
122_collect_lifecycle_results.py
Collect all lifecycle prediction experiment results into a unified summary.

Usage:
  python code/scripts/122_collect_lifecycle_results.py
  python code/scripts/122_collect_lifecycle_results.py --task lifecycle
  python code/scripts/122_collect_lifecycle_results.py --task all
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


def collect_lifecycle_prediction():
    """Collect P0-2 lifecycle prediction results."""
    base = os.path.join(OUT_DIR, "lifecycle_prediction")
    rows = []

    for model in MODELS:
        for h in HORIZONS:
            for s in SEEDS:
                d = os.path.join(base, model, f"h{h}_s{s}_ctx8")
                status_path = os.path.join(d, "run_status.json")
                metrics_path = os.path.join(d, "metrics.json")

                status = "not_started"
                metrics = {}
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        status = json.load(f).get("status", "unknown")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)

                rows.append({
                    "task": "lifecycle",
                    "model": model, "horizon": h, "seed": s,
                    "status": status,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "per_stage_f1": str(metrics.get("per_stage_f1", [])),
                })

    return pd.DataFrame(rows)


def collect_pseudotime_regression():
    """Collect P0-3 pseudotime regression results."""
    base = os.path.join(OUT_DIR, "pseudotime_regression")
    rows = []

    if not os.path.exists(base):
        return pd.DataFrame()

    for model in MODELS:
        for h in HORIZONS:
            for s in SEEDS:
                d = os.path.join(base, model, f"h{h}_s{s}_ctx8")
                status_path = os.path.join(d, "run_status.json")
                metrics_path = os.path.join(d, "metrics.json")

                status = "not_started"
                metrics = {}
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        status = json.load(f).get("status", "unknown")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)

                rows.append({
                    "task": "pseudotime_regression",
                    "model": model, "horizon": h, "seed": s,
                    "status": status,
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "pearson_r": metrics.get("pearson_r"),
                    "spearman_r": metrics.get("spearman_r"),
                    "r2": metrics.get("r2"),
                })

    return pd.DataFrame(rows)


def collect_direction_prediction():
    """Collect P0-4 direction prediction results."""
    base = os.path.join(OUT_DIR, "direction_prediction")
    rows = []

    if not os.path.exists(base):
        return pd.DataFrame()

    for model in MODELS:
        for h in HORIZONS:
            for s in SEEDS:
                d = os.path.join(base, model, f"h{h}_s{s}_ctx8")
                status_path = os.path.join(d, "run_status.json")
                metrics_path = os.path.join(d, "metrics.json")

                status = "not_started"
                metrics = {}
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        status = json.load(f).get("status", "unknown")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)

                rows.append({
                    "task": "direction",
                    "model": model, "horizon": h, "seed": s,
                    "status": status,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "per_class_f1": str(metrics.get("per_class_f1", {})),
                })

    return pd.DataFrame(rows)


def collect_ablation():
    """Collect P0-5 ablation study results."""
    base = os.path.join(OUT_DIR, "ablation")
    rows = []

    if not os.path.exists(base):
        return pd.DataFrame()

    ablation_modes = [
        "concat_fusion", "static_average", "static_gated",
        "pt_only", "horizon_only", "full_pt_horizon", "full_task_embed",
        "rna_only", "protein_only",
    ]

    for mode in ablation_modes:
        for h in HORIZONS:
            for s in SEEDS:
                d = os.path.join(base, mode, f"h{h}_s{s}_ctx8")
                status_path = os.path.join(d, "run_status.json")
                metrics_path = os.path.join(d, "metrics.json")

                status = "not_started"
                metrics = {}
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        status = json.load(f).get("status", "unknown")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)

                rows.append({
                    "task": "ablation",
                    "mode": mode, "horizon": h, "seed": s,
                    "status": status,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                })

    return pd.DataFrame(rows)


def generate_summary(dfs):
    """Generate summary report from all collected dataframes."""
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": {},
    }

    for task_name, df in dfs.items():
        if df.empty:
            report["tasks"][task_name] = {"status": "no_results"}
            continue

        completed = (df["status"] == "completed").sum()
        failed = (df["status"] == "failed").sum()
        total = len(df)

        report["tasks"][task_name] = {
            "total": int(total),
            "completed": int(completed),
            "failed": int(failed),
            "pending": int(total - completed - failed),
            "completion_rate": float(completed / total) if total > 0 else 0,
        }

        # Best results for completed experiments
        if completed > 0 and task_name == "lifecycle":
            done_df = df[df["status"] == "completed"]
            best = done_df.loc[done_df["accuracy"].idxmax()]
            report["tasks"][task_name]["best"] = {
                "model": best["model"],
                "horizon": int(best["horizon"]),
                "accuracy": float(best["accuracy"]),
                "macro_f1": float(best["macro_f1"]),
            }

        if completed > 0 and task_name == "pseudotime_regression":
            done_df = df[df["status"] == "completed"]
            best = done_df.loc[done_df["mae"].idxmin()]
            report["tasks"][task_name]["best"] = {
                "model": best["model"],
                "horizon": int(best["horizon"]),
                "mae": float(best["mae"]),
                "r2": float(best["r2"]),
            }

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["lifecycle", "pseudotime", "direction", "ablation", "all"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Collecting Lifecycle Experiment Results")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    dfs = {}

    if args.task in ("lifecycle", "all"):
        print("\n  Collecting lifecycle prediction...")
        dfs["lifecycle"] = collect_lifecycle_prediction()
        print(f"    {len(dfs['lifecycle'])} experiments")

    if args.task in ("pseudotime", "all"):
        print("\n  Collecting pseudotime regression...")
        dfs["pseudotime_regression"] = collect_pseudotime_regression()
        if not dfs["pseudotime_regression"].empty:
            print(f"    {len(dfs['pseudotime_regression'])} experiments")
        else:
            print("    No results yet")

    if args.task in ("direction", "all"):
        print("\n  Collecting direction prediction...")
        dfs["direction"] = collect_direction_prediction()
        if not dfs["direction"].empty:
            print(f"    {len(dfs['direction'])} experiments")
        else:
            print("    No results yet")

    if args.task in ("ablation", "all"):
        print("\n  Collecting ablation study...")
        dfs["ablation"] = collect_ablation()
        if not dfs["ablation"].empty:
            print(f"    {len(dfs['ablation'])} experiments")
        else:
            print("    No results yet")

    # Save DataFrames
    for name, df in dfs.items():
        if not df.empty:
            csv_path = os.path.join(REPORT_DIR, f"{name}_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n  Saved: {csv_path}")

    # Generate and save summary
    summary = generate_summary(dfs)
    summary_path = os.path.join(REPORT_DIR, "collection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {summary_path}")

    # Print overview
    print("\n" + "=" * 60)
    print("  Overview")
    print("=" * 60)
    for task, info in summary["tasks"].items():
        if "total" in info:
            print(f"  {task}: {info['completed']}/{info['total']} completed "
                  f"({info['completion_rate']:.0%}), {info['failed']} failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
