#!/usr/bin/env python3
"""
108_run_branch_transition_analysis.py
Branch-specific modality transition analysis.
When lineage branches are available, analyze RNA-Protein contribution shifts per branch.

Usage:
  python code/scripts/108_run_branch_transition_analysis.py --branch_method leiden --horizon 4 --seed 42
  python code/scripts/108_run_branch_transition_analysis.py --branch_method diffusion --horizon 4
"""
import os, sys, json
from pathlib import Path, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "branch_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    """Load trajectory sequences and pseudotime."""
    pt_path = os.path.join(PROJECT_ROOT,
        "outputs/highdim_real_mamba/sequences/pbmc_citeseq/rna_hvg1000_protein_concat/pseudotime_window_L32")

    pseudotime = np.load(os.path.join(pt_path, "pseudotime.npy"))
    sequences = np.load(os.path.join(pt_path, "sequences.npy"))

    rna_dim = 1000
    X_rna = sequences[:, :, :rna_dim].astype(np.float32)
    X_prot = sequences[:, :, rna_dim:].astype(np.float32)

    return {
        "X_rna": X_rna, "X_prot": X_prot,
        "pseudotime": pseudotime.astype(np.float32),
        "rna_dim": rna_dim, "prot_dim": sequences.shape[2] - rna_dim,
    }


def assign_branches(data, method="leiden", n_branches=4):
    """Assign branch/cluster labels to trajectory windows.
    Uses the final time step's expression profile for clustering.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # Use mean expression across window as representation
    rna_mean = data["X_rna"].mean(axis=1)  # (N, 1000)

    # PCA reduction
    pca = PCA(n_components=20)
    rna_pca = pca.fit_transform(rna_mean)

    if method in ("kmeans", "leiden"):
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_branches, random_state=42, n_init=10)
        branches = kmeans.fit_predict(rna_pca)
    else:
        # Simple quantile-based branching on pseudotime
        pt_mean = data["pseudotime"].mean(axis=1)
        boundaries = np.quantile(pt_mean, np.linspace(0, 1, n_branches + 1))
        branches = np.digitize(pt_mean, boundaries[1:-1])
        branches = np.clip(branches, 0, n_branches - 1)

    return branches.astype(np.int64)


def compute_modality_weights_per_branch(data, branches, n_branches, n_pt_bins=20):
    """Compute simplified modality contribution weights per branch and pseudotime bin."""
    pt_mean = data["pseudotime"].mean(axis=1)  # (N,)
    rna_var = data["X_rna"].var(axis=1).mean(axis=1)  # (N,) mean variance across genes
    prot_var = data["X_prot"].var(axis=1).mean(axis=1)  # (N,)

    results = {}
    for b in range(n_branches):
        mask = branches == b
        if mask.sum() < 10:
            continue

        pt_b = pt_mean[mask]
        pt_bins = np.digitize(pt_b, np.linspace(pt_b.min(), pt_b.max(), n_pt_bins + 1))
        pt_bins = np.clip(pt_bins, 0, n_pt_bins - 1)

        branch_weights = {}
        for pbin in range(n_pt_bins):
            bin_mask = pt_bins == pbin
            if bin_mask.sum() < 5:
                continue

            rna_v = rna_var[mask][bin_mask].mean()
            prot_v = prot_var[mask][bin_mask].mean()
            total = rna_v + prot_v + 1e-8
            alpha_rna = rna_v / total
            alpha_prot = prot_v / total

            branch_weights[int(pbin)] = {
                "alpha_rna": float(alpha_rna),
                "alpha_prot": float(alpha_prot),
                "n_cells": int(bin_mask.sum()),
            }

        results[f"branch_{b}"] = {
            "n_total": int(mask.sum()),
            "pt_range": [float(pt_b.min()), float(pt_b.max())],
            "weights": branch_weights,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch_method", default="kmeans", choices=["kmeans", "quantile", "leiden"])
    parser.add_argument("--n_branches", type=int, default=4, choices=[2, 3, 4, 5])
    parser.add_argument("--horizon", type=int, default=4, choices=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43, 44])
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Branch Transition Analysis")
    print(f"  Method: {args.branch_method}, Branches: {args.n_branches}")
    print(f"  Horizon: {args.horizon}, Seed: {args.seed}")
    print("=" * 60)

    data = load_data()
    print(f"  Data: RNA {data['X_rna'].shape}, Protein {data['X_prot'].shape}")

    # Assign branches
    branches = assign_branches(data, args.branch_method, args.n_branches)
    unique, counts = np.unique(branches, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Branch {u}: {c} windows ({c/len(branches):.2%})")

    # Compute modality weights per branch
    weights = compute_modality_weights_per_branch(data, branches, args.n_branches)

    # Save results
    result_dir = os.path.join(OUT_DIR, args.branch_method, f"h{args.horizon}_s{args.seed}")
    os.makedirs(result_dir, exist_ok=True)

    results = {
        "branch_method": args.branch_method,
        "n_branches": args.n_branches,
        "horizon": args.horizon,
        "seed": args.seed,
        "branch_distribution": {f"branch_{u}": int(c) for u, c in zip(unique, counts)},
        "modality_weights": weights,
    }

    with open(os.path.join(result_dir, "branch_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Detect transitions
    transitions = []
    for b in range(args.n_branches):
        bk = f"branch_{b}"
        if bk not in weights:
            continue
        w = weights[bk]["weights"]
        for pbin_str, w_data in sorted(w.items(), key=lambda x: int(x[0])):
            pbin = int(pbin_str)
            if pbin > 0 and str(pbin - 1) in w:
                prev_alpha = w[str(pbin - 1)]["alpha_rna"]
                curr_alpha = w_data["alpha_rna"]
                if (prev_alpha > 0.5 and curr_alpha < 0.5) or (prev_alpha < 0.5 and curr_alpha > 0.5):
                    transitions.append({
                        "branch": bk, "pt_bin": pbin,
                        "type": "RNA_to_Protein" if prev_alpha > curr_alpha else "Protein_to_RNA",
                    })

    results["transitions"] = transitions
    with open(os.path.join(result_dir, "branch_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    run_status = {"status": "completed", "n_branches": args.n_branches,
                   "n_transitions": len(transitions), "horizon": args.horizon}
    with open(os.path.join(result_dir, "run_status.json"), "w") as f:
        json.dump(run_status, f, indent=2)

    print(f"\n  Branch analysis complete.")
    print(f"  Transitions detected: {len(transitions)}")
    for t in transitions:
        print(f"    {t['branch']} bin {t['pt_bin']}: {t['type']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
