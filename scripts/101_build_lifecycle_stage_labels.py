#!/usr/bin/env python3
"""
101_build_lifecycle_stage_labels.py
Build lifecycle stage labels from existing pseudotime sequence data.
Reads pseudotime.npy from highdim sequences, discretizes into stages.

Usage:
  # Usage: cd <project_root>
  python code/scripts/101_build_lifecycle_stage_labels.py --bins 4 5 10
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/lifecycle_prediction/lifecycle_labels")
os.makedirs(OUT_DIR, exist_ok=True)

# Sequence directories to process
SEQ_DIRS = [
    "outputs/highdim_real_mamba/sequences/pbmc_citeseq/rna_hvg1000_protein_concat/pseudotime_window_L32",
    "outputs/highdim_real_mamba/sequences/pbmc_citeseq/hvg_1000/pseudotime_window_L32",
    "outputs/highdim_real_mamba/sequences/pbmc_citeseq/pca_300/pseudotime_window_L32",
]


def build_labels_from_pseudotime(pseudotime, n_bins=4, method="quantile"):
    """Build lifecycle stage labels from pseudotime array.
    pseudotime: (N_windows, L) or (N,) array
    Returns: labels of same shape, stage_names dict
    """
    pt = pseudotime.ravel()
    pt_min, pt_max = pt.min(), pt.max()
    pt_norm = (pt - pt_min) / (pt_max - pt_min)

    if method == "quantile":
        boundaries = np.quantile(pt_norm, np.linspace(0, 1, n_bins + 1))
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf

        # digitize returns bin index 1..n, we want 0..n-1
        labels_flat = np.digitize(pt_norm, boundaries[1:-1])
        labels_flat = np.clip(labels_flat, 0, n_bins - 1)
    elif method == "equal_width":
        labels_flat = np.floor(pt_norm * n_bins).astype(int)
        labels_flat = np.clip(labels_flat, 0, n_bins - 1)

    labels = labels_flat.reshape(pseudotime.shape)

    # Stage names
    if n_bins == 4:
        stage_names = {0: "early", 1: "transition", 2: "late", 3: "terminal"}
    elif n_bins == 5:
        stage_names = {0: "early", 1: "early_transition", 2: "transition", 3: "late_transition", 4: "terminal"}
    else:
        stage_names = {i: f"stage_{i}" for i in range(n_bins)}

    return labels, stage_names, float(pt_min), float(pt_max)


def validate_distribution(labels, n_bins):
    """Validate label distribution."""
    labels_flat = labels.ravel()
    results = {
        "n_bins": n_bins,
        "total": int(len(labels_flat)),
        "distribution": {},
        "imbalance_ratio": 0.0,
    }
    counts = []
    for i in range(n_bins):
        c = int((labels_flat == i).sum())
        results["distribution"][f"stage_{i}"] = {"count": c, "fraction": float(c / len(labels_flat))}
        counts.append(c)
    results["imbalance_ratio"] = float(max(counts) / min(counts)) if min(counts) > 0 else -1
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, nargs="+", default=[4, 5, 10])
    parser.add_argument("--method", default="quantile", choices=["quantile", "equal_width"])
    args = parser.parse_args()

    print("=" * 60)
    print("  Lifecycle Stage Label Construction")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}

    for seq_dir_rel in SEQ_DIRS:
        seq_dir = os.path.join(PROJECT_ROOT, seq_dir_rel)
        if not os.path.exists(seq_dir):
            print(f"\n  SKIP: {seq_dir_rel} (not found)")
            continue

        pt_path = os.path.join(seq_dir, "pseudotime.npy")
        if not os.path.exists(pt_path):
            print(f"  SKIP: pseudotime.npy not found in {seq_dir_rel}")
            continue

        pseudotime = np.load(pt_path)
        print(f"\n  Directory: {seq_dir_rel}")
        print(f"  Pseudotime: shape={pseudotime.shape}, range=[{pseudotime.min():.4f}, {pseudotime.max():.4f}]")

        dir_results = {}

        for n_bins in args.bins:
            labels, stage_names, pt_min, pt_max = build_labels_from_pseudotime(pseudotime, n_bins, args.method)

            validation = validate_distribution(labels, n_bins)
            dir_results[f"{n_bins}bin"] = validation

            # Save labels
            dir_name = os.path.basename(seq_dir_rel.replace("/", "_"))
            save_path = os.path.join(OUT_DIR, f"lifecycle_labels_{dir_name}_{n_bins}bin.npy")
            np.save(save_path, labels)

            # Save metadata
            meta = {
                "source": seq_dir_rel,
                "n_bins": n_bins,
                "method": args.method,
                "shape": list(labels.shape),
                "pseudotime_range": [pt_min, pt_max],
                "stage_names": stage_names,
                "distribution": validation["distribution"],
            }
            with open(save_path.replace(".npy", "_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print(f"    {n_bins}bin: saved {save_path}")
            for i in range(n_bins):
                info = validation["distribution"][f"stage_{i}"]
                print(f"      {stage_names[i]}: {info['count']} cells ({info['fraction']:.2%})")
            print(f"      imbalance_ratio: {validation['imbalance_ratio']:.2f}")

        all_results[seq_dir_rel] = dir_results

    # Save overall summary
    summary = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "method": args.method,
        "n_bins": args.bins,
        "results": all_results,
    }
    summary_path = os.path.join(OUT_DIR, "label_construction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    run_status = {
        "status": "completed",
        "task": "lifecycle_label_construction",
        "n_bins_tested": args.bins,
        "n_dirs_processed": len(all_results),
    }
    with open(os.path.join(OUT_DIR, "run_status.json"), "w") as f:
        json.dump(run_status, f, indent=2)

    print("\n  Lifecycle label construction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
