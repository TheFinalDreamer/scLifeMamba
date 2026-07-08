#!/usr/bin/env python3
"""144_rebuild_local_lifecycle_inputs.py — Rebuild all lifecycle pipeline inputs from PBMC CITE-seq h5ad.
Includes: preprocessing, pseudotime, labels, trajectory sequences, direction labels, smoke test.
"""
import json, os, sys, warnings, traceback
from pathlib import Path
from datetime import datetime
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.project_paths import PROJECT_ROOT, DATA_DIR, get_rerun_dir

# === Config ===
SOURCE_H5AD = r"C:\A-KuRuMi\学校专用\数据集存放专用\Hao PBMC multimodal\pbmc_seurat_v4.h5ad"
N_HVG = 1000
WINDOW_SIZE = 32
N_PSEUDOTIME_BINS = 20
SEEDS = [42, 43, 44]
HORIZONS = [1, 2, 4, 8]
LIFECYCLE_BINS = [4, 5]

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SEQ_DIR = PROCESSED_DIR / "trajectory_sequences"
SEQ_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR = PROCESSED_DIR
SPLIT_DIR = PROCESSED_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
DIRECTION_DIR = Path(PROJECT_ROOT) / "outputs" / "revised_direction_labels"
DIRECTION_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_DIR = Path(PROJECT_ROOT) / "outputs" / "local_smoke_test" / datetime.now().strftime("%Y%m%d_%H%M%S")
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = SMOKE_DIR / "smoke_test.log"


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def step1_load_and_preprocess():
    """Load data, QC, normalize, select HVGs, compute PCA/neighbors/UMAP/pseudotime."""
    log("Step 1: Loading PBMC CITE-seq data...")
    import anndata
    import scanpy as sc
    import pandas as pd

    adata = anndata.read_h5ad(SOURCE_H5AD)
    log(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} RNA features")

    # Extract protein data
    protein_df = adata.obsm['protein_counts']  # 161764 x 228 DataFrame
    log(f"  Protein: {protein_df.shape[1]} ADT markers")

    # Basic QC on RNA
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20, :].copy()
    log(f"  After QC: {adata.n_obs} cells x {adata.n_vars} genes")

    # Normalize RNA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    log("  RNA normalized (library-size + log1p)")

    # HVG selection
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat_v3')
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    log(f"  Selected {N_HVG} HVGs")

    # Normalize protein (CLR)
    protein_values = protein_df.values.astype(np.float32)
    protein_clr = protein_values.copy()
    for i in range(protein_clr.shape[1]):
        col = protein_clr[:, i]
        col = col[col > 0]
        if len(col) > 0:
            gm = np.exp(np.mean(np.log(col[col > 0])))
        else:
            gm = 1.0
        protein_clr[:, i] = np.log1p(protein_values[:, i] / (gm + 1e-8))
    protein_clr = np.clip(protein_clr, -10, 10)
    log(f"  Protein CLR normalized: {protein_clr.shape}")

    # PCA
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50, svd_solver='arpack')
    log("  PCA computed (50 components)")

    # Neighbors + UMAP
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata_hvg)
    log("  Neighbors + UMAP computed")

    # Pseudotime (diffusion pseudotime)
    sc.tl.diffmap(adata_hvg)
    adata_hvg.uns['iroot'] = np.flatnonzero(adata_hvg.obsm['X_diffmap'][:, 0].argmax())[0]
    sc.tl.dpt(adata_hvg)
    pseudotime = adata_hvg.obs['dpt_pseudotime'].values.astype(np.float32)
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-8)
    log(f"  Pseudotime computed: range [{pseudotime.min():.4f}, {pseudotime.max():.4f}]")

    return adata_hvg, protein_clr, pseudotime, protein_df


def step2_build_lifecycle_labels(pseudotime, protein_clr):
    """Build lifecycle stage labels from pseudotime quantiles (balanced)."""
    log("Step 2: Building lifecycle stage labels...")
    labels = {}
    for n_bins in LIFECYCLE_BINS:
        # Use quantile-based binning for balanced class distribution
        q_edges = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(pseudotime, q_edges)
        stage_labels = np.digitize(pseudotime, bin_edges[1:-1])
        stage_labels = stage_labels.astype(np.int32)

        # Verify class balance
        unique, counts = np.unique(stage_labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        max_ratio = counts.max() / counts.sum()
        log(f"  {n_bins}-bin labels: distribution={dist}, max_class_ratio={max_ratio:.3f}")

        label_path = LABEL_DIR / f"lifecycle_labels_{n_bins}bin.csv"
        np.savetxt(label_path, stage_labels, fmt='%d', delimiter=',')
        labels[n_bins] = stage_labels

        # Save distribution
        dist_path = DATA_DIR / "metadata" / f"label_distribution_{n_bins}bin.json"
        dist_path.parent.mkdir(parents=True, exist_ok=True)
        save_json({"n_bins": n_bins, "distribution": dist, "max_class_ratio": float(max_ratio)}, dist_path)

    return labels


def step3_build_trajectory_sequences(adata_hvg, protein_clr, pseudotime):
    """Build trajectory window sequences sorted by pseudotime."""
    log("Step 3: Building trajectory sequences...")
    rna_data = adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else (np.array(adata_hvg.X.todense()) if hasattr(adata_hvg.X, 'todense') else np.array(adata_hvg.X))

    # Sort by pseudotime
    sort_idx = np.argsort(pseudotime)
    rna_sorted = rna_data[sort_idx]
    prot_sorted = protein_clr[sort_idx]
    pt_sorted = pseudotime[sort_idx]

    # Concatenate RNA + Protein
    combined = np.concatenate([rna_sorted, prot_sorted], axis=1).astype(np.float32)

    # Build windows of length L
    L = WINDOW_SIZE
    n_windows = len(combined) - L + 1
    sequences = np.array([combined[i:i+L] for i in range(n_windows)])
    pt_windows = np.array([pt_sorted[i:i+L] for i in range(n_windows)])
    log(f"  Sequences: {sequences.shape} (windows x L x features)")
    log(f"  RNA dim: {rna_data.shape[1]}, Protein dim: {protein_clr.shape[1]}, Total: {combined.shape[1]}")

    # Train/val/test split (70/15/15)
    n = len(sequences)
    indices = np.random.RandomState(42).permutation(n)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    splits = {
        "train": indices[:train_end].tolist(),
        "val": indices[train_end:val_end].tolist(),
        "test": indices[val_end:].tolist(),
    }

    # Save
    seq_path = SEQ_DIR / f"trajectory_sequences_L{WINDOW_SIZE}.npz"
    np.savez_compressed(seq_path, sequences=sequences, pseudotime_windows=pt_windows,
                         rna_dim=rna_data.shape[1], protein_dim=protein_clr.shape[1])
    log(f"  Saved: {seq_path}")

    split_path = SPLIT_DIR / f"splits_L{WINDOW_SIZE}.json"
    save_json(splits, split_path)
    log(f"  Splits saved: {split_path}")

    return sequences, pt_windows, splits, sort_idx


def step4_rebuild_direction_labels(pseudotime, sequences, lifecycle_labels_4bin):
    """Rebuild trajectory direction labels (3 schemes)."""
    log("Step 4: Rebuilding trajectory direction labels...")
    results = {}

    # Scheme A: Pseudotime delta direction
    log("  Scheme A: Pseudotime delta direction...")
    # pt_delta per window: pseudotime[-1] - pseudotime[0] for each window
    pt_end = pseudotime[WINDOW_SIZE-1:][:len(sequences)]
    pt_start = pseudotime[:len(sequences)]
    pt_delta = pt_end - pt_start

    # Define thresholds based on quantiles
    low_thresh = np.quantile(pt_delta, 0.3)
    high_thresh = np.quantile(pt_delta, 0.7)
    dir_a = np.zeros(len(pt_delta), dtype=np.int32)
    dir_a[pt_delta < low_thresh] = 0  # backward
    dir_a[(pt_delta >= low_thresh) & (pt_delta <= high_thresh)] = 1  # stationary
    dir_a[pt_delta > high_thresh] = 2  # forward

    dist_a = {int(k): int(v) for k, v in zip(*np.unique(dir_a, return_counts=True))}
    max_a = max(dist_a.values()) / sum(dist_a.values())
    valid_a = max_a <= 0.70
    log(f"    Distribution: {dist_a}, max_ratio={max_a:.3f}, valid={valid_a}")

    # Scheme B: Lifecycle stage transition direction
    log("  Scheme B: Lifecycle stage transition direction...")
    stage_ends = lifecycle_labels_4bin[WINDOW_SIZE-1:len(lifecycle_labels_4bin)]
    stage_starts = lifecycle_labels_4bin[:len(stage_ends)]
    stage_delta = stage_ends - stage_starts

    dir_b = np.zeros(len(stage_delta), dtype=np.int32)  # stable=0
    dir_b[stage_delta > 0] = 1  # forward
    dir_b[stage_delta < 0] = 2  # backward

    dist_b = {int(k): int(v) for k, v in zip(*np.unique(dir_b, return_counts=True))}
    max_b = max(dist_b.values()) / max(sum(dist_b.values()), 1)
    valid_b = max_b <= 0.70
    log(f"    Distribution: {dist_b}, max_ratio={max_b:.3f}, valid={valid_b}")

    # Scheme C: Quantile-adaptive pseudotime delta
    log("  Scheme C: Quantile-adaptive...")
    for q in [0.25, 0.33, 0.40, 0.45]:
        lo, hi = np.quantile(pt_delta, q), np.quantile(pt_delta, 1 - q)
        dir_c = np.zeros(len(pt_delta), dtype=np.int32)
        dir_c[pt_delta < lo] = 0
        dir_c[(pt_delta >= lo) & (pt_delta <= hi)] = 1
        dir_c[pt_delta > hi] = 2
        dist_c = {int(k): int(v) for k, v in zip(*np.unique(dir_c, return_counts=True))}
        max_c = max(dist_c.values()) / sum(dist_c.values())
        if max_c <= 0.70:
            log(f"    q={q}: distribution={dist_c}, max_ratio={max_c:.3f}, VALID")
            break
        log(f"    q={q}: distribution={dist_c}, max_ratio={max_c:.3f}, invalid")

    valid_c = max_c <= 0.70

    # Save all schemes
    schemes = {
        "A_pseudotime_delta": {"labels": dir_a.tolist(), "distribution": dist_a, "max_ratio": max_a, "valid": valid_a},
        "B_lifecycle_transition": {"labels": dir_b.tolist(), "distribution": dist_b, "max_ratio": max_b, "valid": valid_b},
        "C_quantile_adaptive": {"labels": dir_c.tolist(), "distribution": dist_c, "max_ratio": max_c, "valid": valid_c if 'max_c' in dir() else False},
    }

    # Overwrite C
    schemes["C_quantile_adaptive"]["valid"] = valid_c

    # Select best
    best = None
    for name, scheme in schemes.items():
        if scheme["valid"]:
            if best is None or scheme["max_ratio"] < best["max_ratio"]:
                best = {"scheme": name, **scheme}

    quality = {
        "old_scheme": {"name": "lifecycle_stage_delta_32step", "max_ratio": 0.9987, "valid": False,
                       "note": "99.87% single class — FAILED"},
        "schemes": schemes,
        "best": best,
        "recommendation": best["scheme"] if best else "all_failed",
    }

    # Save
    for name, scheme in schemes.items():
        csv_path = DIRECTION_DIR / f"direction_labels_{name}.csv"
        np.savetxt(csv_path, scheme["labels"], fmt='%d', delimiter=',')
    save_json(quality, DIRECTION_DIR / "label_quality_report.json")

    if best:
        csv_path = DIRECTION_DIR / "selected_direction_labels.csv"
        np.savetxt(csv_path, best["labels"], fmt='%d', delimiter=',')
        log(f"  Best scheme: {best['scheme']} (max_ratio={best['max_ratio']:.3f})")
    else:
        log("  WARNING: No valid direction label scheme found!")

    # Write README
    readme = [
        "# Revised Trajectory Direction Labels",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Old Scheme (FAILED)",
        "- lifecycle_stage_delta over 32-step window",
        "- 99.87% stationary class → EXCLUDED from valid results",
        "",
        "## New Schemes",
    ]
    for name, s in schemes.items():
        readme.append(f"### {name}")
        readme.append(f"- Distribution: {s['distribution']}")
        readme.append(f"- Max class ratio: {s['max_ratio']:.3f}")
        readme.append(f"- Valid: {s['valid']}")
    if best:
        readme.append(f"\n## Selected: {best['scheme']}")
        readme.append(f"- Max class ratio: {best['max_ratio']:.3f}")
    (DIRECTION_DIR / "README.md").write_text('\n'.join(readme), encoding='utf-8')

    return quality


def step5_smoke_test_lifecycle_prediction(sequences, labels_4bin, splits):
    """Run minimal lifecycle prediction smoke test (MLP, CPU, 2 epochs)."""
    log("Step 5: Smoke test — Lifecycle Prediction (MLP, 2 epochs, CPU)...")
    import torch
    import torch.nn as nn

    # Prepare tiny subset
    n_train = min(200, len(splits["train"]))
    n_val = min(50, len(splits["val"]))
    train_idx = splits["train"][:n_train]
    val_idx = splits["val"][:n_val]

    X_train = torch.tensor(sequences[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels_4bin[train_idx], dtype=torch.long)
    X_val = torch.tensor(sequences[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels_4bin[val_idx], dtype=torch.long)

    # Mean pool the sequence
    X_train_pooled = X_train.mean(dim=1)
    X_val_pooled = X_val.mean(dim=1)

    input_dim = X_train_pooled.shape[1]
    n_classes = 4

    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, n_classes)
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    log(f"  Training: {n_train} train, {n_val} val, input_dim={input_dim}, classes={n_classes}")
    for epoch in range(2):
        model.train()
        opt.zero_grad()
        logits = model(X_train_pooled)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_pooled)
            val_loss = loss_fn(val_logits, y_val)
            val_pred = val_logits.argmax(dim=1)
            acc = (val_pred == y_val).float().mean().item()
        log(f"    Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, val_acc={acc:.4f}")

    # Save smoke test results
    smoke_results = {
        "task": "lifecycle_prediction_smoke_test",
        "model": "MLP (2-layer)",
        "epochs": 2,
        "n_train": n_train,
        "n_val": n_val,
        "final_val_acc": acc,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "is_real_mamba": False,
        "is_fallback": True,
        "is_smoke_test": True,
        "can_be_used_in_paper": False,
        "note": "CPU smoke test with fallback MLP only. NOT a paper result.",
        "timestamp": datetime.now().isoformat(),
    }
    save_json(smoke_results, SMOKE_DIR / "metrics.json")
    save_json({"status": "completed", "type": "smoke_test", "timestamp": datetime.now().isoformat()},
              SMOKE_DIR / "run_status.json")
    save_json({"epochs": 2, "model": "mlp", "device": "cpu", "is_real_mamba": False},
              SMOKE_DIR / "config.json")

    return smoke_results


def step6_smoke_test_direction(sequences, direction_labels, splits, scheme_name):
    """Smoke test for direction prediction."""
    log(f"Step 6: Smoke test — Direction Prediction ({scheme_name})...")
    import torch
    import torch.nn as nn

    n_train = min(200, len(splits["train"]))
    n_val = min(50, len(splits["val"]))
    train_idx = splits["train"][:n_train]
    val_idx = splits["val"][:n_val]

    X_train = torch.tensor(sequences[train_idx], dtype=torch.float32).mean(dim=1)
    y_train = torch.tensor(np.array(direction_labels)[train_idx], dtype=torch.long)
    X_val = torch.tensor(sequences[val_idx], dtype=torch.float32).mean(dim=1)
    y_val = torch.tensor(np.array(direction_labels)[val_idx], dtype=torch.long)

    n_classes = len(set(direction_labels))
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64), nn.ReLU(),
        nn.Linear(64, n_classes)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(dim=1)
            acc = (val_pred == y_val).float().mean().item()
        log(f"    Epoch {epoch+1}: loss={loss.item():.4f}, val_acc={acc:.4f}")

    dir_results = {
        "task": "direction_prediction_smoke_test",
        "scheme": scheme_name,
        "n_classes": n_classes,
        "final_val_acc": acc,
        "is_smoke_test": True,
        "can_be_used_in_paper": False,
    }
    save_json(dir_results, DIRECTION_DIR / "direction_smoke_test_metrics.json")
    return dir_results


def main():
    log("=" * 60)
    log("144: Rebuilding Local Lifecycle Pipeline Inputs")
    log(f"Source: {SOURCE_H5AD}")
    log(f"Environment: CPU, fallback Mamba only")
    log("=" * 60)

    manifest = {"timestamp": datetime.now().isoformat(), "steps": {}}

    try:
        # Step 1
        adata_hvg, protein_clr, pseudotime, protein_df = step1_load_and_preprocess()
        manifest["steps"]["load_and_preprocess"] = {"status": "ok", "n_cells": adata_hvg.n_obs, "n_hvg": adata_hvg.n_vars}

        # Step 2
        labels = step2_build_lifecycle_labels(pseudotime, protein_clr)
        manifest["steps"]["lifecycle_labels"] = {"status": "ok", "bins": list(labels.keys())}

        # Step 3
        sequences, pt_windows, splits, sort_idx = step3_build_trajectory_sequences(adata_hvg, protein_clr, pseudotime)
        manifest["steps"]["trajectory_sequences"] = {"status": "ok", "n_sequences": len(sequences), "window_size": WINDOW_SIZE}

        # Step 4
        direction_quality = step4_rebuild_direction_labels(pseudotime, sequences, labels[4])
        manifest["steps"]["direction_labels"] = {"status": "ok", "best_scheme": direction_quality.get("best", {}).get("scheme", "none")}

        # Step 5
        smoke_results = step5_smoke_test_lifecycle_prediction(sequences, labels[4], splits)
        manifest["steps"]["lifecycle_smoke_test"] = {"status": "ok", "val_acc": smoke_results["final_val_acc"]}

        # Step 6 — if valid direction scheme exists
        if direction_quality.get("best"):
            best_scheme = direction_quality["best"]
            dir_results = step6_smoke_test_direction(sequences, best_scheme["labels"], splits, best_scheme["scheme"])
            manifest["steps"]["direction_smoke_test"] = {"status": "ok", "scheme": best_scheme["scheme"]}
        else:
            manifest["steps"]["direction_smoke_test"] = {"status": "skipped", "reason": "no valid scheme"}
            log("  Skipping direction smoke test: no valid scheme")

        # Save manifests
        save_json(manifest, DATA_DIR / "metadata" / "input_manifest.json")
        qr = {
            "n_cells": adata_hvg.n_obs,
            "n_rna_features": adata_hvg.n_vars,
            "n_protein_features": protein_clr.shape[1],
            "n_sequences": len(sequences),
            "window_size": WINDOW_SIZE,
            "lifecycle_bins": list(labels.keys()),
            "pseudotime_range": [float(pseudotime.min()), float(pseudotime.max())],
            "direction_schemes_tested": 3,
            "direction_best_scheme": direction_quality.get("best", {}).get("scheme", "none"),
            "smoke_test_passed": True,
        }
        save_json(qr, DATA_DIR / "metadata" / "data_quality_report.json")

        log("\n" + "=" * 60)
        log("ALL STEPS COMPLETE")
        log(f"Smoke test output: {SMOKE_DIR}")
        log(f"Direction labels: {DIRECTION_DIR}")
        log(f"Processed data: {PROCESSED_DIR}")
        log("=" * 60)

    except Exception as e:
        log(f"FATAL ERROR: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        manifest["status"] = "failed"
        manifest["error"] = str(e)
        save_json(manifest, DATA_DIR / "metadata" / "input_manifest.json")


if __name__ == "__main__":
    main()
