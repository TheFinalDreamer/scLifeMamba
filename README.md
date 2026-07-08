# scLifeMamba

Trajectory-aware Mamba-LSTM framework for single-cell multi-omics lifecycle state modeling.

## Overview

scLifeMamba encodes pseudotime-ordered multi-omics sequences (RNA + protein) using a gated Mamba-LSTM architecture with pseudotime- and horizon-aware dynamic fusion (LagAwareDynamicFusion). The framework supports lifecycle stage prediction, future pseudotime regression, trajectory direction analysis, and modality contribution quantification.

Key features:

- **Mamba-LSTM encoder**: Selective state-space scanning combined with bidirectional LSTM recurrence for trajectory sequence modeling.
- **LagAwareDynamicFusion**: Pseudotime- and horizon-conditioned gating that adaptively weights RNA and protein modalities.
- **Dual backend**: Native Mamba SSM kernel (`mamba-ssm`) for production use; fallback Conv1d+GRU block for development and CPU-only environments.
- **Modality analysis**: Systematic comparison of RNA-only, protein-only, and fused modality configurations for lifecycle phenotype characterization.
- **Revised direction labels**: Identifies and corrects class-imbalance failures in naive pseudotime-delta trajectory direction labeling.

## Repository structure

```
scLifeMamba/
├── src/                    # Core model and utility modules
│   ├── models/             # scMultiLifeMamba, MambaBlock, LagAwareDynamicFusion, etc.
│   └── utils/              # Path resolution, config, I/O, seed utilities
├── scripts/                # Experiment and preprocessing scripts (101-144 series)
├── configs/                # YAML configuration files
├── examples/               # Minimal smoke test and example config
├── tests/                  # Unit and integration tests
├── docs/                   # Method overview, data preparation, reproducibility
├── figures/                # README figures only
└── data/                   # Data preparation guide (no real data included)
```

## Installation

```bash
git clone https://github.com/TheFinalDreamer/scLifeMamba.git
cd scLifeMamba
pip install -r requirements.txt
```

For native Mamba support (optional, requires Linux with CUDA):

```bash
pip install mamba-ssm causal-conv1d
```

The codebase functions without `mamba-ssm` using the built-in fallback backend for development, debugging, and CPU-only workflows.

## Data preparation

This repository does **not** include single-cell data files. Users must obtain the PBMC CITE-seq dataset from Hao et al. (2021), available through the Seurat v4 reference resources.

After downloading:

```bash
# Prepare the PBMC CITE-seq data
python scripts/143_prepare_pbmc_citeseq_data.py

# Rebuild lifecycle inputs (labels, trajectory windows)
python scripts/144_rebuild_local_lifecycle_inputs.py

# Build trajectory direction labels
python scripts/130_rebuild_trajectory_direction_labels.py
```

See `data/README.md` for detailed data preparation instructions.

## Quick start

A minimal smoke test is provided to verify installation:

```bash
python examples/minimal_smoke_test.py
```

This test uses synthetic data and does not require GPU or `mamba-ssm`.

## Reproducing experiments

### Preprocessing (CPU, no GPU required)

```bash
python scripts/09_compute_pseudotime.py
python scripts/101_build_lifecycle_stage_labels.py
python scripts/13_build_trajectory_sequences.py
```

### Lifecycle prediction

```bash
python scripts/102_run_future_lifecycle_prediction.py --config configs/lifecycle_prediction.yaml
```

Supports models: `mlp`, `lstm`, `transformer`, `mamba`, `mamba_lstm`, `lag_aware_fusion`.  
Horizons: 1, 2, 4, 8. Seeds: 42, 43, 44.

### Pseudotime regression

```bash
python scripts/103_run_future_pseudotime_regression.py --config configs/pseudotime_regression.yaml
```

### Ablation study

```bash
python scripts/105_run_ablation_study.py --config configs/ablation.yaml
```

### Protein dominance analysis

```bash
python scripts/133_analyze_protein_dominance.py
```

### Revised trajectory direction prediction

```bash
python scripts/131_run_revised_trajectory_direction_prediction.py --config configs/direction.yaml
```

## Notes on Mamba backend

The project supports two backends:

| Backend | Kernel | Environment | Use case |
|---------|--------|-------------|----------|
| Native | `mamba-ssm` | Linux + CUDA | Production training, full experiments |
| Fallback | Conv1d + GRU | Any (CPU/GPU, Windows/Linux) | Development, debugging, smoke tests |

The fallback backend provides a functional approximation but does not implement selective state-space scanning. When reporting results, clearly indicate which backend was used.

## Output files

Each experiment run produces:

```
outputs/<experiment_name>/<timestamp>/
├── run_status.json       # Experiment completion status
├── metrics.json          # Classification/regression metrics
├── config.json           # Runtime configuration
└── predictions.csv       # Model predictions (when applicable)
```

## Citation

If you use scLifeMamba in your research, please cite:

```bibtex
@software{scLifeMamba2026,
  author       = {Du, Wendong},
  title        = {scLifeMamba: Trajectory-aware Mamba-LSTM framework for single-cell multi-omics lifecycle state modeling},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/TheFinalDreamer/scLifeMamba}
}
```

See `CITATION.cff` for the full metadata record.

## License

MIT License. See `LICENSE` for details.
