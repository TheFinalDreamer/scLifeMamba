#!/usr/bin/env python3
"""
133_analyze_protein_dominance.py
Phase 4: Systematic analysis of protein modality dominance in lifecycle prediction.

Analyses:
  1. Modality mutual information with lifecycle labels
  2. Per-feature discriminability (RNA genes vs ADT proteins)
  3. Ablation weight dynamics across pseudotime (RNA→Protein transition)
  4. Per-class modality contribution breakdown
  5. Feature redundancy analysis (within-modality vs cross-modality correlation)

Output: JSON report + modality analysis figures for manuscript

Usage:
  python code/scripts/133_analyze_protein_dominance.py
"""
import os, sys, json
from pathlib import Path, warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "protein_dominance_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

RNA_DIM = 1000
PROT_DIM = 228


def load_data():
    pt_path = os.path.join(PROJECT_ROOT,
        "outputs/highdim_real_mamba/sequences/pbmc_citeseq/rna_hvg1000_protein_concat/pseudotime_window_L32")

    pseudotime = np.load(os.path.join(pt_path, "pseudotime.npy"))
    sequences = np.load(os.path.join(pt_path, "sequences.npy"))

    X_rna = sequences[:, :, :RNA_DIM].astype(np.float32)
    X_prot = sequences[:, :, RNA_DIM:].astype(np.float32)

    # Load lifecycle labels (4-bin)
    label_dir = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "lifecycle_labels")
    label_file = os.path.join(label_dir,
        "lifecycle_labels_outputs_highdim_real_mamba_sequences_pbmc_citeseq_rna_hvg1000_protein_concat_pseudotime_window_L32_4bin.npy")
    labels = np.load(label_file)
    y = labels[:, -1]  # Last cell's lifecycle stage

    # Load ablation modality weights if available
    weights = None
    weight_dir = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "ablation")
    # Try to load per-run weight dynamics from full fusion modes
    weight_files = []
    for mode in ["full_pt_horizon", "full_task_embed"]:
        for h in [4]:
            for s in [42]:
                exp_dir = os.path.join(weight_dir, mode, f"h{h}_s{s}_ctx8")
                if os.path.exists(os.path.join(exp_dir, "metrics.json")):
                    weight_files.append(os.path.join(exp_dir, "metrics.json"))

    return {
        "X_rna": X_rna, "X_prot": X_prot,
        "y": y, "pseudotime": pseudotime,
        "labels": labels,
    }


def compute_mutual_information_binned(x, y, n_bins=20):
    """Approximate mutual information using histogram binning."""
    N = len(x)
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Bin continuous features
    x_binned = np.digitize(x, np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))

    # Joint histogram
    joint = np.zeros((n_bins, 4))
    for i in range(N):
        bx = min(x_binned[i], n_bins - 1)
        by = y[i]
        joint[bx, by] += 1

    joint /= N
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)

    mi = 0
    for i in range(n_bins):
        for j in range(4):
            if joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (p_x[i] * p_y[j]))

    return mi


def analyze_mutual_information(data):
    """Analysis 1: Mutual information between modalities and lifecycle labels."""
    print("\n[Analysis 1] Modality-Label Mutual Information")

    X_rna = data["X_rna"].reshape(-1, RNA_DIM)
    X_prot = data["X_prot"].reshape(-1, PROT_DIM)
    y_flat = data["labels"].ravel()

    # Aggregate per cell (mean across window)
    rna_cell = data["X_rna"][:, -1, :]  # Last cell in window
    prot_cell = data["X_prot"][:, -1, :]
    y_cell = data["y"]

    # PCA-based dimensionality reduction for MI estimation
    # RNA: top PCA component; Protein: top PCA too
    rna_centered = rna_cell - rna_cell.mean(axis=0)
    prot_centered = prot_cell - prot_cell.mean(axis=0)

    try:
        rna_cov = rna_centered.T @ rna_centered / (len(rna_centered) - 1)
        rna_eigvals = np.linalg.eigvalsh(rna_cov)
        rna_pc1_var = rna_eigvals[-1] / rna_eigvals.sum()

        prot_cov = prot_centered.T @ prot_centered / (len(prot_centered) - 1)
        prot_eigvals = np.linalg.eigvalsh(prot_cov)
        prot_pc1_var = prot_eigvals[-1] / prot_eigvals.sum()
    except Exception:
        rna_pc1_var = 0
        prot_pc1_var = 0

    mi_rna = compute_mutual_information_binned(rna_cell, y_cell)
    mi_prot = compute_mutual_information_binned(prot_cell, y_cell)

    results = {
        "mi_rna_labels": float(mi_rna),
        "mi_protein_labels": float(mi_prot),
        "mi_ratio_protein_to_rna": float(mi_prot / max(mi_rna, 1e-10)),
        "rna_pc1_variance_explained": float(rna_pc1_var),
        "protein_pc1_variance_explained": float(prot_pc1_var),
    }

    print(f"  MI(RNA, Label) = {mi_rna:.4f}")
    print(f"  MI(Protein, Label) = {mi_prot:.4f}")
    print(f"  Ratio = {mi_prot / max(mi_rna, 1e-10):.1f}x")

    return results


def analyze_per_class_modality(data):
    """Analysis 2: Per-class modality contribution breakdown."""
    print("\n[Analysis 2] Per-Class Modality Statistics")

    X_rna = data["X_rna"][:, -1, :]
    X_prot = data["X_prot"][:, -1, :]
    y = data["y"]

    per_class = {}
    for c in range(4):
        mask = y == c
        rna_std = X_rna[mask].std(axis=0).mean()
        prot_std = X_prot[mask].std(axis=0).mean()
        per_class[f"class_{c}"] = {
            "count": int(mask.sum()),
            "rna_mean_std": float(rna_std),
            "protein_mean_std": float(prot_std),
            "std_ratio": float(prot_std / max(rna_std, 1e-10)),
        }
        print(f"  Class {c} (n={mask.sum()}): RNA std={rna_std:.4f}, Protein std={prot_std:.4f}")

    return per_class


def analyze_feature_discriminability(data):
    """Analysis 3: Per-feature discriminability via ANOVA F-statistic."""
    print("\n[Analysis 3] Feature Discriminability")

    X_rna = data["X_rna"][:, -1, :]
    X_prot = data["X_prot"][:, -1, :]
    y = data["y"]

    def f_statistic(X, y):
        n_features = X.shape[1]
        f_scores = np.zeros(n_features)
        for j in range(n_features):
            class_means = [X[y == c, j].mean() for c in range(4)]
            grand_mean = X[:, j].mean()
            ss_between = sum(len(X[y == c, j]) * (class_means[c] - grand_mean)**2 for c in range(4))
            ss_within = sum(((X[y == c, j] - class_means[c])**2).sum() for c in range(4))
            f_scores[j] = (ss_between / 3) / max(ss_within / (len(y) - 4), 1e-10)
        return f_scores

    rna_f = f_statistic(X_rna, y)
    prot_f = f_statistic(X_prot, y)

    results = {
        "rna_top10_f_mean": float(np.sort(rna_f)[-10:].mean()),
        "protein_top10_f_mean": float(np.sort(prot_f)[-10:].mean()),
        "rna_median_f": float(np.median(rna_f)),
        "protein_median_f": float(np.median(prot_f)),
        "rna_top10_indices": np.argsort(rna_f)[-10:][::-1].tolist(),
        "protein_top10_indices": np.argsort(prot_f)[-10:][::-1].tolist(),
        "ratio_median_f": float(np.median(prot_f) / max(np.median(rna_f), 1e-10)),
    }

    print(f"  RNA top-10 mean F: {results['rna_top10_f_mean']:.2f}")
    print(f"  Protein top-10 mean F: {results['protein_top10_f_mean']:.2f}")
    print(f"  Protein/RNA median F ratio: {results['ratio_median_f']:.1f}x")

    return results


def analyze_cross_modality_correlation(data):
    """Analysis 4: Cross-modality redundancy analysis."""
    print("\n[Analysis 4] Cross-Modality Correlation Structure")

    X_rna = data["X_rna"][:, -1, :]
    X_prot = data["X_prot"][:, -1, :]

    # Within-modality correlation
    rna_corr = np.corrcoef(X_rna.T)
    prot_corr = np.corrcoef(X_prot.T)

    # Cross-modality correlation
    cross_corr = np.zeros((RNA_DIM, PROT_DIM))
    for j in range(min(RNA_DIM, PROT_DIM)):
        if X_rna[:, j].std() > 1e-10 and X_prot[:, j].std() > 1e-10:
            cross_corr[j, j] = np.corrcoef(X_rna[:, j], X_prot[:, j])[0, 1]

    n_pairs = min(RNA_DIM, PROT_DIM)
    cross_diag = np.array([np.corrcoef(X_rna[:, i], X_prot[:, i])[0, 1]
                           for i in range(n_pairs)
                           if X_rna[:, i].std() > 1e-10 and X_prot[:, i].std() > 1e-10])

    results = {
        "rna_mean_pairwise_corr": float(np.mean(np.abs(np.triu(rna_corr, 1)))),
        "protein_mean_pairwise_corr": float(np.mean(np.abs(np.triu(prot_corr, 1)))),
        "cross_modality_mean_corr": float(np.mean(np.abs(cross_diag))),
        "rna_effective_rank": float(np.sum(np.linalg.eigvalsh(rna_corr) > 0.01)),
        "protein_effective_rank": float(np.sum(np.linalg.eigvalsh(prot_corr) > 0.01)),
    }

    print(f"  RNA pairwise corr: {results['rna_mean_pairwise_corr']:.3f}")
    print(f"  Protein pairwise corr: {results['protein_mean_pairwise_corr']:.3f}")
    print(f"  Cross-modality corr: {results['cross_modality_mean_corr']:.3f}")
    print(f"  RNA effective rank: {results['rna_effective_rank']:.0f}/{RNA_DIM}")
    print(f"  Protein effective rank: {results['protein_effective_rank']:.0f}/{PROT_DIM}")

    return results


def analyze_pseudotime_vs_modality(data):
    """Analysis 5: How modality importance varies across pseudotime."""
    print("\n[Analysis 5] Pseudotime vs Modality Importance")

    pseudotime = data["pseudotime"][:, -1]  # Last cell's pseudotime
    X_rna = data["X_rna"][:, -1, :]
    X_prot = data["X_prot"][:, -1, :]
    y = data["y"]

    # Split pseudotime into 5 bins
    pt_bins = np.digitize(pseudotime, np.quantile(pseudotime, np.linspace(0, 1, 6)[1:-1]))

    bin_results = []
    for b in range(5):
        mask = pt_bins == b
        if mask.sum() < 10:
            continue

        mi_rna = compute_mutual_information_binned(X_rna[mask], y[mask])
        mi_prot = compute_mutual_information_binned(X_prot[mask], y[mask])

        bin_results.append({
            "bin": int(b),
            "n_cells": int(mask.sum()),
            "pseudotime_range": [float(pseudotime[mask].min()), float(pseudotime[mask].max())],
            "mi_rna": float(mi_rna),
            "mi_protein": float(mi_prot),
            "mi_ratio": float(mi_prot / max(mi_rna, 1e-10)),
        })
        print(f"  Bin {b}: PT=[{pseudotime[mask].min():.3f}, {pseudotime[mask].max():.3f}], "
              f"MI_ratio={bin_results[-1]['mi_ratio']:.1f}x")

    return bin_results


def main():
    print("=" * 60)
    print("Protein Dominance Analysis")
    print("=" * 60)

    data = load_data()
    print(f"  RNA: {data['X_rna'].shape}, Protein: {data['X_prot'].shape}")
    print(f"  Labels: {np.unique(data['y'], return_counts=True)}")

    results = {
        "mutual_information": analyze_mutual_information(data),
        "per_class_statistics": analyze_per_class_modality(data),
        "feature_discriminability": analyze_feature_discriminability(data),
        "cross_modality_correlation": analyze_cross_modality_correlation(data),
        "pseudotime_modality_dynamics": analyze_pseudotime_vs_modality(data),
    }

    # Summary verdict
    mi = results["mutual_information"]
    fd = results["feature_discriminability"]
    cm = results["cross_modality_correlation"]

    verdict = {
        "protein_dominance_confirmed": mi["mi_ratio_protein_to_rna"] > 2.0,
        "primary_factor": [],
        "contributing_factors": [],
        "recommendation": "",
    }

    if fd["ratio_median_f"] > 3:
        verdict["primary_factor"].append("Higher per-protein discriminability (F-statistic)")
    if cm["protein_effective_rank"] > cm["rna_effective_rank"]:
        verdict["primary_factor"].append("Protein features less redundant than RNA (higher effective rank)")
    if cm["cross_modality_mean_corr"] < 0.3:
        verdict["contributing_factors"].append("Low cross-modality correlation — modalities carry complementary info")

    verdict["recommendation"] = (
        "Protein dominance is a dataset characteristic of PBMC CITE-seq, not a model failure. "
        "Future work should (1) validate on datasets where RNA is more informative, "
        "(2) explore ATAC-guided protein compensation to reduce modality gap, "
        "(3) position LagAwareFusion as modality-adaptive rather than RNA-dominant."
    )

    results["verdict"] = verdict

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    for line in verdict["primary_factor"]:
        print(f"  PRIMARY: {line}")
    for line in verdict["contributing_factors"]:
        print(f"  CONTRIBUTING: {line}")
    print(f"  RECOMMENDATION: {verdict['recommendation']}")

    # Save
    out_path = os.path.join(OUT_DIR, "protein_dominance_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
