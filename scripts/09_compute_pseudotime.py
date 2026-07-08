#!/usr/bin/env python3
"""09_compute_pseudotime.py — Compute DPT pseudotime for trajectory datasets.

Input: standardized h5ad
Output: h5ad with dpt_pseudotime, metadata JSON, optional UMAP plot.
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np


def compute_pseudotime(input_path, output_path, metadata_path, fig_path=None, seed=42):
    """Compute DPT pseudotime on standardized AnnData."""
    import scanpy as sc
    import anndata

    print("Loading: " + str(input_path))
    adata = anndata.read_h5ad(input_path)

    # Check for existing pseudotime
    existing = None
    for col in adata.obs.columns:
        if any(kw in col.lower() for kw in ["dpt", "pseudotime", "palantir", "latent_time"]):
            existing = col
            break

    if existing:
        print("Using existing pseudotime field: " + existing)
        pseudotime_key = existing
        method = "pre_existing"
        root_info = "Pre-existing pseudotime column: " + existing
    else:
        print("Computing DPT pseudotime...")
        method = "dpt"

        # PCA
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50, random_state=seed)

        # Neighbors
        sc.pp.neighbors(adata, n_neighbors=15, random_state=seed)

        # Diffusion map
        sc.tl.diffmap(adata, random_state=seed)

        # DPT root selection
        if "paul15_clusters" in adata.obs:
            labels = adata.obs["paul15_clusters"]
            root_idx = int(np.argmin(adata.obs["paul15_clusters"].cat.codes.values))
            root_cell = str(adata.obs.index[root_idx])
            root_info = "Auto-selected root from earliest cluster index (cell: {})".format(root_cell)
        else:
            root_idx = 0
            root_cell = str(adata.obs.index[0])
            root_info = "Heuristic root: first cell (index 0)"

        adata.uns["iroot"] = root_idx
        sc.tl.dpt(adata)
        pseudotime_key = "dpt_pseudotime"
        print("DPT computed. Root: " + root_info)

    # UMAP for visualization
    if fig_path:
        try:
            if "X_umap" not in adata.obsm:
                sc.tl.umap(adata, random_state=seed)
            sc.pl.umap(adata, color=pseudotime_key, show=False,
                       save="_pseudotime", title="DPT Pseudotime")
            print("UMAP figure saved")
        except Exception as e:
            print("UMAP figure failed (non-blocking): " + str(e))

    # Save output
    adata.write_h5ad(output_path)
    print("Saved: " + str(output_path))

    # Metadata
    meta = {
        "dataset": Path(input_path).parent.name,
        "method": method,
        "pseudotime_key": pseudotime_key,
        "root_selection": root_info,
        "n_cells": int(adata.n_obs),
        "pseudotime_range": [float(adata.obs[pseudotime_key].min()),
                             float(adata.obs[pseudotime_key].max())],
        "pseudotime_mean": float(adata.obs[pseudotime_key].mean()),
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "DPT-derived pseudotime used as weak-supervision trajectory label. "
            "This is NOT equivalent to real experimental time."
        ),
    }
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("Metadata: " + str(metadata_path))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return compute_pseudotime(args.input, args.output, args.metadata, args.figure, args.seed)


if __name__ == "__main__":
    sys.exit(main())
