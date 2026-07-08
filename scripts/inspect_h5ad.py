#!/usr/bin/env python
"""Inspect an h5ad file and generate a data report.

Usage:
    python code/scripts/inspect_h5ad.py --h5ad_path path/to/data.h5ad
    python code/scripts/inspect_h5ad.py --h5ad_path path/to/data.h5ad --output_dir code/output/data_reports
"""

import argparse
import os
import sys
import json
import numpy as np
from datetime import datetime
from scipy.sparse import issparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _check_anndata():
    """Check if anndata is installed, provide install instructions if not."""
    try:
        import anndata
        return anndata
    except ImportError:
        print("=" * 60)
        print("ERROR: anndata is not installed.")
        print("Install with: pip install anndata scanpy")
        print("=" * 60)
        sys.exit(1)


def _safe_len(x):
    """Get length without converting full sparse matrix."""
    try:
        return len(x)
    except Exception:
        return "unknown"


def inspect_h5ad(h5ad_path: str, output_dir: str = None):
    """Inspect an h5ad file and generate a markdown report."""
    anndata = _check_anndata()

    if not os.path.exists(h5ad_path):
        print(f"ERROR: File not found: {h5ad_path}")
        sys.exit(1)

    print(f"Loading {h5ad_path} ...")
    adata = anndata.read_h5ad(h5ad_path)

    # Prepare output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join("code", "output", "data_reports")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{ts}_h5ad_inspection_report.md")

    lines = []
    def w(text=""):
        lines.append(text)

    w(f"# h5ad Data Inspection Report")
    w()
    w(f"- **File**: `{h5ad_path}`")
    w(f"- **Date**: {ts}")
    w(f"- **AnnData version**: {anndata.__version__}")
    w()

    # Basic info
    w("## 1. Basic Information")
    w()
    w(f"- **Shape**: {adata.n_obs} cells x {adata.n_vars} genes")
    w(f"- **X dtype**: {adata.X.dtype}")
    w(f"- **Sparse**: {issparse(adata.X)}")
    w()

    # Data range
    if issparse(adata.X):
        w(f"- **X min**: {adata.X.data.min():.4f}")
        w(f"- **X max**: {adata.X.data.max():.4f}")
        w(f"- **X mean**: {adata.X.data.mean():.4f}")
        w(f"- **Sparsity**: {(adata.X.nnz / (adata.n_obs * adata.n_vars) * 100):.2f}%")
    else:
        w(f"- **X min**: {np.min(adata.X):.4f}")
        w(f"- **X max**: {np.max(adata.X):.4f}")
        w(f"- **X mean**: {np.mean(adata.X):.4f}")
    w()

    # Missing values
    w("## 2. Missing Value Check")
    w()
    if issparse(adata.X):
        nan_count = np.isnan(adata.X.data).sum()
    else:
        nan_count = np.isnan(adata.X).sum()
    w(f"- **NaN count in X**: {nan_count}")
    if nan_count > 0:
        w(f"  - WARNING: {nan_count} NaN values detected!")
    else:
        w(f"  - OK: No NaN values.")
    w()

    # obs columns
    w("## 3. Observations (obs) Columns")
    w()
    w(f"Total obs columns: {len(adata.obs.columns)}")
    w()
    for col in adata.obs.columns[:20]:
        col_dtype = adata.obs[col].dtype
        n_unique = adata.obs[col].nunique()
        n_missing = adata.obs[col].isna().sum()
        extra = ""
        if col_dtype.name == "category":
            cats = list(adata.obs[col].cat.categories)
            if len(cats) <= 10:
                extra = f"categories: {cats}"
        w(f"- `{col}`: dtype={col_dtype}, unique={n_unique}, missing={n_missing} {extra}")
    if len(adata.obs.columns) > 20:
        w(f"- ... and {len(adata.obs.columns) - 20} more columns")
    w()

    # Label key recommendations
    w("## 4. Label Key Recommendations")
    w()
    label_candidates = []
    for col in adata.obs.columns:
        n_unique = adata.obs[col].nunique()
        if 2 <= n_unique <= 50 and adata.obs[col].dtype.name in ("category", "object"):
            label_candidates.append((col, n_unique))

    if label_candidates:
        w("Candidate label keys (categorical, 2-50 classes):")
        w()
        for col, n in sorted(label_candidates, key=lambda x: x[1]):
            w(f"- `{col}`: {n} classes")
            # Class distribution
            vc = adata.obs[col].value_counts()
            for cls_name, cnt in vc.items():
                pct = cnt / adata.n_obs * 100
                w(f"  - {cls_name}: {cnt} ({pct:.1f}%)")
    else:
        w("No suitable categorical label key found with 2-50 classes.")
    w()

    # var columns
    w("## 5. Variables (var) Columns")
    w()
    w(f"Total var columns: {len(adata.var.columns)}")
    w()
    for col in list(adata.var.columns)[:10]:
        w(f"- `{col}`: dtype={adata.var[col].dtype}")
    if len(adata.var.columns) > 10:
        w(f"- ... and {len(adata.var.columns) - 10} more columns")
    w()

    # obsm keys
    w("## 6. obsm Keys (Multi-modal Data)")
    w()
    if adata.obsm:
        for key in adata.obsm:
            shape = adata.obsm[key].shape
            w(f"- `{key}`: shape={shape}, dtype={adata.obsm[key].dtype}")
    else:
        w("- No obsm keys found.")
    w()

    # Protein key recommendations
    w("## 7. Protein obsm Key Recommendations")
    w()
    protein_candidates = [k for k in adata.obsm if "protein" in k.lower()]
    if protein_candidates:
        w("Candidate protein keys:")
        for k in protein_candidates:
            w(f"- `{k}`: shape={adata.obsm[k].shape}")
    else:
        w("No protein-like obsm key found. If this is RNA-only data, set `use_protein: false`.")
    w()

    # layers
    w("## 8. Layers")
    w()
    if adata.layers:
        for key in adata.layers:
            s = adata.layers[key].shape
            w(f"- `{key}`: shape={s}")
    else:
        w("- No layers found.")
    w()

    # uns
    w("## 9. Unstructured (uns) Keys")
    w()
    for key in sorted(adata.uns.keys()):
        val = adata.uns[key]
        val_type = type(val).__name__
        w(f"- `{key}`: type={val_type}")
    w()

    # Pseudotime check
    w("## 10. Pseudotime Check")
    w()
    ptime_candidates = [c for c in adata.obs.columns if "pseudotime" in c.lower() or "ptime" in c.lower() or "dpt" in c.lower()]
    if ptime_candidates:
        w("Candidate pseudotime columns:")
        for c in ptime_candidates:
            v = adata.obs[c].values
            w(f"- `{c}`: min={np.min(v):.4f}, max={np.max(v):.4f}, dtype={adata.obs[c].dtype}")
    else:
        w("No pseudotime-like column found. Set `use_pseudotime: false` or compute pseudotime separately.")
    w()

    # Summary checklist
    w("## 11. Recommended Config")
    w()
    w("```yaml")
    w("data:")
    w(f"  h5ad_path: \"{h5ad_path}\"")
    if label_candidates:
        w(f"  label_key: \"{label_candidates[0][0]}\"  # or choose from recommendations above")
    if protein_candidates:
        w(f"  protein_obsm_key: \"{protein_candidates[0]}\"")
        w(f"  use_protein: true")
    else:
        w(f"  use_protein: false")
    if ptime_candidates:
        w(f"  pseudotime_key: \"{ptime_candidates[0]}\"")
        w(f"  use_pseudotime: true")
    else:
        w(f"  use_pseudotime: false")
    w(f"  normalize: true")
    w(f"  log1p: true")
    w("```")

    # Write report
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Inspect an h5ad file for scLifeMamba compatibility")
    parser.add_argument("--h5ad_path", type=str, required=True, help="Path to h5ad file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for report")
    args = parser.parse_args()

    inspect_h5ad(args.h5ad_path, args.output_dir)


if __name__ == "__main__":
    main()
