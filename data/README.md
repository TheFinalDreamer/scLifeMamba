# Data directory

This repository does **not** include raw or processed single-cell data files.

## Required dataset

PBMC CITE-seq multimodal dataset from Hao et al. (2021):

- **Source**: Seurat v4 reference resources
- **Reference**: Hao, Y. et al. Integrated analysis of multimodal single-cell data. *Cell* 184, 3573–3587 (2021).
- **Contents**: 161,764 PBMCs, 20,729 RNA genes, 228 ADT proteins

## Obtaining the data

1. Download from the Seurat v4 multimodal reference:
   - Visit https://satijalab.org/seurat/
   - Or use `SeuratData` R package: `InstallData("pbmc multimodal")`
2. Convert to h5ad format for Python/Scanpy compatibility
3. Place the file at `data/raw/pbmc_citeseq.h5ad` or set `SCLIFEMAMBA_DATA` environment variable

## Preprocessing

Run the following scripts in order:

```bash
# 1. Locate and validate the downloaded data
python scripts/143_prepare_pbmc_citeseq_data.py

# 2. Compute diffusion pseudotime
python scripts/09_compute_pseudotime.py

# 3. Build lifecycle stage labels (4-bin quantile)
python scripts/101_build_lifecycle_stage_labels.py

# 4. Construct trajectory sequences (L=32 windows)
python scripts/13_build_trajectory_sequences.py

# 5. Rebuild local lifecycle inputs (recovery workflow)
python scripts/144_rebuild_local_lifecycle_inputs.py

# 6. Build revised trajectory direction labels
python scripts/130_rebuild_trajectory_direction_labels.py
```

## Output structure

Successful preprocessing produces:

```
data/
├── raw/
│   └── pbmc_citeseq.h5ad          # User-provided data file (not in repo)
├── processed/
│   ├── lifecycle_labels_4bin.npy
│   ├── trajectory_sequences.npy
│   ├── pseudotime.npy
│   └── direction_labels_schemeB.npy
└── metadata/
    ├── input_manifest.json
    └── data_quality_report.json
```

## Notes

- Do not commit large data files (`.h5ad`, `.npy`, `.h5`) to GitHub.
- Use the `SCLIFEMAMBA_DATA` environment variable to specify a custom data directory.
- See `docs/DATA_PREPARATION.md` for detailed instructions.
