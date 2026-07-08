#!/usr/bin/env python3
"""13_build_trajectory_sequences.py — Build pseudotime-window cell-state sequences.

Loads RNA data and DPT pseudotime, computes PCA, then builds trajectory-aware
sequences by sliding a window over pseudotime-ordered cells.

Output (all saved to --output-dir):
  features_pca.npy                        — (n_cells, n_components) PCA features
  sequences_pseudotime_window_L16.npy      — (n_cells, 16, n_components) sequences
  labels.npy                               — (n_cells,) cell-type labels
  pseudotime.npy                           — (n_cells,) DPT pseudotime values
  cell_indices.npy                         — (n_cells,) original cell indices
  sequence_metadata.json                   — parameters and shape info
  pseudotime_order.csv                     — cell ordering for inspection

Usage:
  python code/scripts/13_build_trajectory_sequences.py \
    --dataset paul15 \
    --project-root /data/scLifeMamba \
    --feature-mode pca --n-components 50 \
    --sequence-length 16 --sequence-mode pseudotime_window \
    --overwrite
"""

import sys
import os
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_sequences(args):
    """Main entry point."""
    root = Path(args.project_root)
    input_dir = root / "outputs/processed" / args.dataset
    output_dir = root / "outputs/phase6/trajectory_sequences" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_file = output_dir / "sequences_pseudotime_window_L{}.npy".format(args.sequence_length)
    meta_file = output_dir / "sequence_metadata.json"

    if seq_file.exists() and not args.overwrite:
        print("Sequences already exist: " + str(seq_file))
        print("Use --overwrite to rebuild.")
        return 0

    # ── 1. Load RNA data ──────────────────────────────────────────────
    rna_path = input_dir / "rna.npy"
    if not rna_path.exists():
        rna_path = input_dir / "x_rna.npy"
    if not rna_path.exists():
        raise FileNotFoundError("RNA data not found in " + str(input_dir))
    rna = np.load(rna_path).astype(np.float32)
    print("RNA shape: " + str(rna.shape))

    # ── 2. Load labels ────────────────────────────────────────────────
    label_path = input_dir / "labels.npy"
    if not label_path.exists():
        label_path = input_dir / "cell_types.npy"
    if label_path.exists():
        labels = np.load(label_path).astype(np.int64)
    else:
        labels = np.zeros(len(rna), dtype=np.int64)
    print("Labels shape: " + str(labels.shape))

    # ── 3. Load pseudotime ────────────────────────────────────────────
    ptime_path = args.pseudotime
    if ptime_path is None:
        ptime_path = input_dir / "pseudotime.npy"
    ptime_path = Path(ptime_path)
    if not ptime_path.exists():
        raise FileNotFoundError("Pseudotime not found: " + str(ptime_path))
    pseudotime = np.load(str(ptime_path)).astype(np.float32)
    # Normalize to [0, 1]
    pmin, pmax = pseudotime.min(), pseudotime.max()
    if pmax > pmin:
        pseudotime = (pseudotime - pmin) / (pmax - pmin)
    print("Pseudotime shape: {}, range: [{:.4f}, {:.4f}]".format(
        len(pseudotime), pseudotime.min(), pseudotime.max()))

    n_cells = len(rna)
    n_genes = rna.shape[1]
    print("Cells: {}, Genes: {}".format(n_cells, n_genes))

    # ── 4. PCA compression ────────────────────────────────────────────
    n_components = min(args.n_components, n_cells, n_genes)
    print("Computing PCA: {} components...".format(n_components))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = StandardScaler()
        rna_scaled = scaler.fit_transform(rna)
        pca = PCA(n_components=n_components, random_state=args.seed)
        features_pca = pca.fit_transform(rna_scaled).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    print("PCA features shape: {}".format(features_pca.shape))
    print("Explained variance ({:.0f} PCs): {:.4f}".format(n_components, explained))

    # ── 5. Sort by pseudotime ─────────────────────────────────────────
    sort_idx = np.argsort(pseudotime)
    features_sorted = features_pca[sort_idx]
    labels_sorted = labels[sort_idx]
    pseudotime_sorted = pseudotime[sort_idx]

    # Save pseudotime order for inspection
    order_csv = output_dir / "pseudotime_order.csv"
    with open(str(order_csv), "w") as f:
        f.write("pseudotime_order_idx,original_idx,pseudotime,label\n")
        for i in range(n_cells):
            f.write("{},{},{:.6f},{}\n".format(
                i, sort_idx[i], pseudotime_sorted[i], labels_sorted[i]))

    # ── 6. Build pseudotime-window sequences ──────────────────────────
    L = args.sequence_length
    half = L // 2
    n_feat = n_components
    sequences = np.zeros((n_cells, L, n_feat), dtype=np.float32)

    for i in range(n_cells):
        start = i - half
        end = i + (L - half)
        indices = np.arange(start, end)
        # Clamp to valid range
        indices = np.clip(indices, 0, n_cells - 1)
        sequences[i] = features_sorted[indices]

    print("Sequences shape: {}".format(sequences.shape))
    print("Boundary cells with repeated entries: {}".format(
        sum(1 for i in range(n_cells) if i - half < 0 or i + (L - half) > n_cells)))

    # ── 7. Save outputs ───────────────────────────────────────────────
    np.save(str(output_dir / "features_pca.npy"), features_pca)
    np.save(str(output_dir / "sequences_pseudotime_window_L{}.npy".format(L)), sequences)
    np.save(str(output_dir / "labels.npy"), labels)
    np.save(str(output_dir / "pseudotime.npy"), pseudotime)
    np.save(str(output_dir / "cell_indices.npy"), np.arange(n_cells, dtype=np.int32))

    metadata = {
        "dataset": args.dataset,
        "n_cells": n_cells,
        "n_genes_original": n_genes,
        "feature_mode": args.feature_mode,
        "n_components": n_components,
        "pca_explained_variance": float(explained),
        "sequence_mode": args.sequence_mode,
        "sequence_length": L,
        "sequence_shape": list(sequences.shape),
        "pseudotime_source": str(ptime_path),
        "pseudotime_range_after_norm": [float(pseudotime.min()), float(pseudotime.max())],
        "boundary_description": (
            "Pad-to-L: for cells near boundaries, window is clamped to [0, n_cells-1]. "
            "Boundary cells may have repeated entries at the edges."
        ),
        "pseudotime_order_file": str(order_csv),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
    }
    with open(str(meta_file), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("Done. Outputs in: " + str(output_dir))
    print("  features_pca.npy:               " + str(features_pca.shape))
    print("  sequences_pseudotime_window_L{}.npy: ".format(L) + str(sequences.shape))
    print("  metadata:                       " + str(meta_file))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Build pseudotime-window cell-state sequences for Phase 6"
    )
    parser.add_argument("--dataset", default="paul15")
    parser.add_argument("--project-root", default="/data/scLifeMamba")
    parser.add_argument("--pseudotime", default=None,
                        help="Path to pseudotime.npy (default: outputs/processed/<dataset>/pseudotime.npy)")
    parser.add_argument("--feature-mode", default="pca", choices=["pca"])
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-mode", default="pseudotime_window")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return build_sequences(args)


if __name__ == "__main__":
    sys.exit(main())
