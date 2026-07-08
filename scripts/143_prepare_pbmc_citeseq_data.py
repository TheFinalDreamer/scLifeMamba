#!/usr/bin/env python3
"""143_prepare_pbmc_citeseq_data.py — Locate and validate PBMC CITE-seq data source."""
import json, csv, os, sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.project_paths import PROJECT_ROOT, DATA_DIR, get_recovery_dir

CANDIDATE_PATHS = [
    r"C:\A-KuRuMi\学校专用\数据集存放专用\Hao PBMC multimodal\pbmc_seurat_v4.h5ad",
    r"C:\A-KuRuMi\学校专用\数据集存放专用\pbmc10k_raw.h5ad",
    r"C:\A-KuRuMi\学校专用\数据集存放专用\10X_3-rep2.h5ad",
    r"C:\A-KuRuMi\学校专用\数据集存放专用\Human PBMC Glaucoma Atlas.h5ad",
]

OUT_DIR = Path(PROJECT_ROOT) / "outputs" / "local_data_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_candidate(path_str):
    path = Path(path_str)
    if not path.exists():
        return {"path": path_str, "status": "not_found"}

    info = {
        "path": path_str,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "status": "found",
        "has_rna": False,
        "has_protein": False,
        "has_celltype": False,
        "has_pseudotime": False,
        "has_split": False,
        "n_cells": 0,
        "n_rna_features": 0,
        "n_protein_features": 0,
        "usable_for_lifecycle_pipeline": False,
        "score": 0,
    }

    try:
        import anndata
        adata = anndata.read_h5ad(path_str)
        info["n_cells"] = adata.n_obs
        info["n_rna_features"] = adata.n_vars
        info["has_rna"] = True
        info["score"] += 20

        # Check for protein
        if hasattr(adata, 'obsm') and 'protein_counts' in adata.obsm:
            pc = adata.obsm['protein_counts']
            info["n_protein_features"] = pc.shape[1]
            info["has_protein"] = True
            info["score"] += 30

        # Check for cell type labels
        type_cols = [c for c in adata.obs.columns if 'celltype' in c.lower() or 'cell_type' in c.lower()]
        if type_cols:
            info["has_celltype"] = True
            info["score"] += 10

        # Check for pseudotime
        pt_cols = [c for c in adata.obs.columns if 'pseudotime' in c.lower() or 'dpt' in c.lower()]
        if pt_cols:
            info["has_pseudotime"] = True
            info["score"] += 15

        # Usability check
        if info["has_rna"] and info["has_protein"] and info["n_cells"] > 50000:
            info["usable_for_lifecycle_pipeline"] = True
            info["score"] += 25
            info["status"] = "selected"
        elif info["has_rna"] and info["n_cells"] > 10000:
            info["status"] = "rna_only_usable"
        else:
            info["status"] = "insufficient"

    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)[:200]

    return info


def main():
    print("=== PBMC CITE-seq Data Search ===")
    results = []
    for path in CANDIDATE_PATHS:
        info = inspect_candidate(path)
        results.append(info)
        flag = "[OK]" if info.get("usable_for_lifecycle_pipeline") else "[--]"
        print(f"  {flag} {Path(path).name}: {info['status']} | cells={info['n_cells']} | RNA={info['n_rna_features']} | Protein={info['n_protein_features']} | score={info['score']}")

    # Find best candidate
    best = max(results, key=lambda r: r["score"])
    print(f"\nBest candidate: {Path(best['path']).name} (score={best['score']})")

    # Save results
    csv_path = OUT_DIR / "pbmc_citeseq_candidates.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "candidates": results,
        "selected": best if best.get("usable_for_lifecycle_pipeline") else None,
        "selected_path": best["path"],
        "selected_n_cells": best["n_cells"],
        "selected_n_protein": best["n_protein_features"],
    }
    json_path = OUT_DIR / "pbmc_citeseq_candidates.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Save manifest
    manifest_path = DATA_DIR / "metadata" / "pbmc_citeseq_data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest saved: {manifest_path}")
    print(f"Candidates CSV: {csv_path}")
    print(f"Candidates JSON: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
