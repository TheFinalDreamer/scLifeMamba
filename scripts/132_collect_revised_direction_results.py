#!/usr/bin/env python3
"""
132_collect_revised_direction_results.py
Collect and aggregate results from the revised trajectory direction prediction (script 131).
Generates summary CSV and JSON for manuscript integration.

Usage:
  python code/scripts/132_collect_revised_direction_results.py
"""
import os, sys, json
from pathlib import Path, glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "revised_direction_prediction")
COLLECT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "collected")
os.makedirs(COLLECT_DIR, exist_ok=True)

MODELS = ["mlp", "lstm", "transformer", "mamba", "mamba_lstm", "lag_aware_fusion"]


def collect_results():
    rows = []
    for model in MODELS:
        model_dir = os.path.join(OUT_DIR, model)
        if not os.path.exists(model_dir):
            print(f"  WARN: {model} directory not found")
            continue

        for exp_dir in sorted(glob.glob(os.path.join(model_dir, "h*_s*_ctx*"))):
            metrics_path = os.path.join(exp_dir, "metrics.json")
            status_path = os.path.join(exp_dir, "run_status.json")

            if not os.path.exists(metrics_path):
                continue

            with open(status_path) as f:
                status = json.load(f)

            with open(metrics_path) as f:
                metrics = json.load(f)

            rows.append({
                "model": model,
                "horizon": status["horizon"],
                "seed": status["seed"],
                "context_window": status.get("context_window", 8),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "per_class_f1": metrics.get("per_class_f1", {}),
                "confusion_matrix": metrics.get("confusion_matrix", []),
                "status": status.get("status", "unknown"),
            })

    if not rows:
        print("ERROR: No results found")
        return

    df = pd.DataFrame(rows)

    # Aggregate by model (mean/std across horizons and seeds)
    agg = df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        n_runs=("accuracy", "count"),
    ).reset_index()

    agg = agg.sort_values("macro_f1_mean", ascending=False)

    # Save
    detail_csv = os.path.join(COLLECT_DIR, "revised_direction_detail.csv")
    summary_csv = os.path.join(COLLECT_DIR, "revised_direction_summary.csv")
    summary_json = os.path.join(COLLECT_DIR, "revised_direction_summary.json")

    df.to_csv(detail_csv, index=False)
    agg.to_csv(summary_csv, index=False)

    json_out = {
        "models": agg.to_dict(orient="records"),
        "best_model": {
            "name": agg.iloc[0]["model"],
            "macro_f1": float(agg.iloc[0]["macro_f1_mean"]),
            "accuracy": float(agg.iloc[0]["accuracy_mean"]),
        },
        "n_total_experiments": len(rows),
        "n_models_tested": len(MODELS),
    }
    with open(summary_json, "w") as f:
        json.dump(json_out, f, indent=2)

    print("=" * 60)
    print("Revised Direction Prediction — Aggregated Results")
    print("=" * 60)
    print(agg.to_string(index=False))
    print(f"\n  Best: {json_out['best_model']['name']} (F1={json_out['best_model']['macro_f1']:.4f})")
    print(f"  Saved to: {summary_json}")


if __name__ == "__main__":
    collect_results()
