#!/usr/bin/env python
"""Preprocess a real h5ad dataset for scLifeMamba training.

Usage:
    python code/scripts/preprocess_dataset.py --config code/configs/dataset/pbmc_citeseq.yaml
    python code/scripts/preprocess_dataset.py --input data/raw/dataset.h5ad --output data/processed/
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.io import save_json, ensure_dir
from src.data.preprocessing import (
    preprocess_rna,
    preprocess_protein,
    encode_labels,
    build_data_summary,
)


def save_array(data: np.ndarray, dirpath: str, name: str, fmt: str = "npz"):
    """Save numpy array in npz (compressed) or npy format."""
    if fmt == "npz":
        np.savez_compressed(os.path.join(dirpath, f"{name}.npz"), data)
    else:
        np.save(os.path.join(dirpath, f"{name}.npy"), data)


def main():
    parser = argparse.ArgumentParser(description="Preprocess h5ad dataset for scLifeMamba")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset YAML config (e.g., configs/dataset/pbmc_citeseq.yaml)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input h5ad file (overrides config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--label_key", type=str, default=None, help="obs key for labels")
    parser.add_argument("--protein_key", type=str, default="protein", help="obsm key for protein")
    parser.add_argument("--pseudotime_key", type=str, default=None, help="obs key for pseudotime")
    parser.add_argument("--batch_key", type=str, default=None, help="obs key for batch")
    parser.add_argument("--use_protein", action="store_true", default=None, help="Enable protein modality")
    parser.add_argument("--no_protein", dest="use_protein", action="store_false", help="Disable protein (RNA-only)")
    parser.add_argument("--max_cells", type=int, default=None, help="Max cells to load")
    parser.add_argument("--hvg_top_k", type=int, default=None, help="Top HVG to keep")
    parser.add_argument("--save_format", type=str, default="npz", choices=["npz", "npy"])
    parser.set_defaults(use_protein=None)
    args = parser.parse_args()

    # Load config if provided
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        cfg = cfg.get("data", cfg)
    elif args.config:
        print(f"Warning: config file not found: {args.config}")

    # Merge args into config (args take precedence)
    h5ad_path = args.input or cfg.get("h5ad_path", "")
    if not h5ad_path or not os.path.exists(h5ad_path):
        print("ERROR: No h5ad file specified. Use --input or set h5ad_path in config.")
        sys.exit(1)

    output_root = args.output or cfg.get("processed_dir", os.path.join("code", "data", "processed"))
    label_key = args.label_key or cfg.get("label_key", "cell_type")
    protein_key = args.protein_key or cfg.get("protein_obsm_key", "protein")
    pseudotime_key = args.pseudotime_key or cfg.get("pseudotime_key", "pseudotime")
    batch_key = args.batch_key or cfg.get("batch_key", None)
    use_protein = args.use_protein if args.use_protein is not None else cfg.get("use_protein", True)
    max_cells = args.max_cells or cfg.get("max_cells", None)
    hvg_top_k = args.hvg_top_k or cfg.get("hvg_top_k", None)
    save_format = args.save_format or cfg.get("save_format", "npz")
    normalize = cfg.get("normalize", True)
    log1p = cfg.get("log1p", True)
    scale = cfg.get("scale", False)

    train_ratio = cfg.get("train_ratio", 0.7)
    val_ratio = cfg.get("val_ratio", 0.15)
    test_ratio = cfg.get("test_ratio", 0.15)
    seed = cfg.get("seed", 42)

    # Load h5ad
    try:
        import anndata
    except ImportError:
        print("ERROR: anndata is required. Install with: pip install anndata scanpy")
        sys.exit(1)

    print(f"Loading {h5ad_path} ...")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Subsample
    if max_cells and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"Subsampled to {max_cells} cells")

    # Extract RNA
    from scipy.sparse import issparse
    rna_layer = cfg.get("rna_layer", None)
    if rna_layer and rna_layer in adata.layers:
        x_rna = adata.layers[rna_layer]
        print(f"Using RNA layer: {rna_layer}")
    else:
        x_rna = adata.X

    if issparse(x_rna):
        x_rna = x_rna.toarray()
    x_rna = np.asarray(x_rna, dtype=np.float32)
    print(f"RNA shape: {x_rna.shape}")

    # Extract Protein
    if use_protein:
        if protein_key in adata.obsm:
            x_protein = np.asarray(adata.obsm[protein_key], dtype=np.float32)
            print(f"Protein shape: {x_protein.shape}")
        else:
            print(f"WARNING: protein key '{protein_key}' not found in adata.obsm. Switching to RNA-only.")
            use_protein = False
            x_protein = np.zeros((x_rna.shape[0], 1), dtype=np.float32)
    else:
        x_protein = np.zeros((x_rna.shape[0], 1), dtype=np.float32)

    # Labels
    if label_key in adata.obs:
        labels, label_mapping = encode_labels(adata.obs[label_key].values)
        print(f"Label key: {label_key}, classes: {len(label_mapping)}")
        for i, name in label_mapping.items():
            cnt = (labels == int(i)).sum()
            print(f"  Class {i} ({name}): {cnt} cells")
    else:
        available = list(adata.obs.columns)
        print(f"ERROR: label_key '{label_key}' not found in adata.obs. Available: {available}")
        sys.exit(1)

    # Pseudotime (optional)
    pseudotime = None
    use_pseudotime = cfg.get("use_pseudotime", False)
    ptime_key = pseudotime_key or cfg.get("pseudotime_key", None)
    if use_pseudotime and ptime_key and ptime_key in adata.obs:
        pseudotime = adata.obs[ptime_key].values.astype(np.float32)
        pmin, pmax = pseudotime.min(), pseudotime.max()
        if pmax > pmin:
            pseudotime = (pseudotime - pmin) / (pmax - pmin)
        print(f"Pseudotime range: [{pmin:.4f}, {pmax:.4f}] -> normalized to [0, 1]")
    elif use_pseudotime:
        print(f"WARNING: Pseudotime key '{ptime_key}' not found. Disabling pseudotime.")
        use_pseudotime = False

    # Batch
    batch = None
    if batch_key and batch_key in adata.obs:
        batch_vals, _ = encode_labels(adata.obs[batch_key].values)
        batch = batch_vals.astype(np.int64)
        print(f"Batch key: {batch_key}, batches: {len(np.unique(batch))}")

    # Preprocess
    print("Preprocessing...")
    x_rna = preprocess_rna(x_rna, normalize=normalize, log1p=log1p, hvg_top_k=hvg_top_k)
    if use_protein and x_protein.shape[1] > 1:
        x_protein = preprocess_protein(x_protein, normalize=normalize, log1p=log1p)
    print(f"After preprocessing - RNA: {x_rna.shape}, Protein: {x_protein.shape}")

    # Train/val/test split
    from src.data.split import stratified_split
    train_idx, val_idx, test_idx = stratified_split(
        labels, train_ratio=train_ratio, val_ratio=val_ratio,
        test_ratio=test_ratio, seed=seed,
    )
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Build data summary
    data_summary = build_data_summary(
        x_rna=x_rna, x_protein=x_protein, labels=labels,
        pseudotime=pseudotime, label_mapping=label_mapping,
        batch=batch, use_protein=use_protein, use_pseudotime=use_pseudotime,
    )

    # Create output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = cfg.get("name", "dataset")
    out_dir = os.path.join(output_root, f"{ts}_{dataset_name}")
    ensure_dir(out_dir)

    # Save data
    print(f"Saving to {out_dir} ...")
    save_array(x_rna, out_dir, "x_rna", fmt=save_format)
    if use_protein and x_protein.shape[1] > 1:
        save_array(x_protein, out_dir, "x_protein", fmt=save_format)
    save_array(labels, out_dir, "labels", fmt=save_format)
    if pseudotime is not None:
        save_array(pseudotime, out_dir, "pseudotime", fmt=save_format)
    if batch is not None:
        save_array(batch, out_dir, "batch", fmt=save_format)

    # Save metadata
    save_json(label_mapping, os.path.join(out_dir, "label_mapping.json"))
    save_json(data_summary, os.path.join(out_dir, "data_summary.json"))
    split_indices = {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}
    save_json(split_indices, os.path.join(out_dir, "split_indices.json"))

    # Save config used
    cfg_out = {
        "h5ad_path": h5ad_path,
        "label_key": label_key,
        "protein_key": protein_key,
        "pseudotime_key": pseudotime_key,
        "batch_key": batch_key,
        "use_protein": use_protein,
        "use_pseudotime": use_pseudotime,
        "max_cells": max_cells,
        "hvg_top_k": hvg_top_k,
        "normalize": normalize,
        "log1p": log1p,
        "scale": scale,
        "save_format": save_format,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }
    save_json(cfg_out, os.path.join(out_dir, "preprocess_config_used.json"))

    print(f"\nPreprocessing complete!")
    print(f"Output directory: {out_dir}")
    print(f"Files saved: {os.listdir(out_dir)}")
    print(f"\nTo use this data, set in experiment config:")
    print(f"  data:")
    print(f"    mode: processed")
    print(f'    processed_dir: "{out_dir}"')


if __name__ == "__main__":
    main()
