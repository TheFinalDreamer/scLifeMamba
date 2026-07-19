# Reproducibility

This repository is aligned with the manuscript code availability statement.

## Environment

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Pandas
- Scanpy
- AnnData
- scikit-learn

Install with:

```bash
pip install -r requirements.txt
```

## Data

Raw PBMC CITE-seq data are not redistributed. The expected processed directory is:

```text
data/processed/leakage_safe_v1/
```

Expected files:

```text
features_combined.npy
rna_hvg_all.npy
protein_norm_all.npy
sequence_manifest.csv
reference_sequence_manifest.parquet
split_assignments.csv
hvg_genes_train_only.txt
```

## Split And Preprocessing

The manuscript uses a donor-held-out split:

- 5 training donors
- 2 validation donors
- 1 test donor

All preprocessing steps are fit on training donors only, including HVG selection, scaling, PCA, nearest-neighbor graph construction, diffusion pseudotime fitting, and lifecycle thresholding.

## Backend

The reported Mamba backend is `TorchSelectiveSSM`, implemented in `src/models/torch_selective_ssm.py`.

`mamba_ssm` is not required for the manuscript results.

## Commands

```bash
python scripts/run_final_audit_experiments.py --exp fair --epochs 10
python scripts/run_final_audit_experiments.py --exp order --epochs 10
python scripts/81_compute_baselines.py
```

Outputs are written under `outputs/`, which is intentionally ignored by Git.
