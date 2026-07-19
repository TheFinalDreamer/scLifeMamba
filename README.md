# scLifeMamba

Trajectory-aware Mamba-LSTM framework for multimodal single-cell state modeling under donor-held-out leakage-controlled evaluation.

This repository accompanies the manuscript:

`scLifeMamba: Trajectory-Aware Mamba-LSTM Framework for Multimodal Single-Cell State Modeling`

## Architecture

scLifeMamba combines:

- RNA and protein encoders for per-cell multimodal features
- Dynamic modality fusion over RNA and protein channels
- A Mamba-style selective state-space sequence block
- A 2-layer unidirectional LSTM for local transition refinement
- A classification head for lifecycle state prediction

The Mamba backend used for the manuscript is `TorchSelectiveSSM`.

This repository provides a PyTorch implementation of selective state-space modeling used in the manuscript. It does not require the Linux-only `mamba_ssm` CUDA package.

## Results Summary

The manuscript evaluates 161,764 PBMC CITE-seq cells from the Seurat v4 reference dataset under a donor-held-out protocol with 5 training donors, 2 validation donors, and 1 test donor.

The leakage-safe preprocessing pipeline produces 643,384 target-isolated sequences across horizons 1, 4, 8, and 16.

Fair backbone comparison in the manuscript uses 3 seeds, 10 epochs, an 8k subset, and horizon 1:

| Model | Macro F1 | Accuracy | Parameters |
|---|---:|---:|---:|
| MLP mean-pool | 0.9349 +/- 0.0149 | 0.9435 +/- 0.0061 | 763,783 |
| LSTM-only | 0.9232 +/- 0.0082 | 0.9256 +/- 0.0067 | 1,044,743 |
| Mamba-only | 0.9275 +/- 0.0029 | 0.9234 +/- 0.0139 | 897,287 |
| Mamba-LSTM | 0.9278 +/- 0.0053 | 0.9352 +/- 0.0016 | 1,161,480 |

The MLP mean-pooling model has a larger F1 in this specific setting. The Mamba-LSTM contribution is the trajectory-aware architecture, leakage-controlled evaluation workflow, and reproducible selective state-space implementation.

## Installation

```bash
git clone https://github.com/TheFinalDreamer/scLifeMamba.git
cd scLifeMamba

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS users can activate the environment with:

```bash
source .venv/bin/activate
```

Conda users can use:

```bash
conda env create -f environment.yml
conda activate sclifemamba
```

## Reproducibility

### Dataset Preparation

The PBMC CITE-seq dataset is from the Seurat v4 reference resource:

```bash
wget https://atlas.fredhutch.org/data/nygc/multimodal/pbmc_multimodal.h5seurat
```

Raw single-cell data are not redistributed in this repository. Place processed leakage-safe inputs under:

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

### Donor-Held-Out Split

The manuscript uses a donor-held-out split:

- 5 donors for training
- 2 donors for validation
- 1 donor for testing

Preprocessing is fit on training donors only. This includes highly variable gene selection, scaling, PCA, nearest-neighbor graph construction, diffusion pseudotime fitting, and lifecycle thresholding. Validation and test cells are mapped into the train-fitted preprocessing space.

### TorchSelectiveSSM Implementation

`src/models/torch_selective_ssm.py` implements the selective state-space model in native PyTorch. The implementation includes input-dependent delta, B, and C projections, a learned stable A parameterization, and selective recurrence. The manuscript reports this backend explicitly as `TorchSelectiveSSM`, not native `mamba_ssm`.

## Training

Quick single-run command:

```bash
python scripts/run_mamba_final_experiments.py \
    --exp backbone \
    --model mamba_lstm \
    --horizon 1 \
    --seed 42 \
    --epochs 10 \
    --batch_size 64
```

Fair backbone comparison:

```bash
python scripts/run_final_audit_experiments.py \
    --exp fair \
    --epochs 10 \
    --lr 1e-3
```

Sequence order ablation:

```bash
python scripts/run_final_audit_experiments.py \
    --exp order \
    --epochs 10 \
    --lr 1e-3
```

Classical baseline matrix:

```bash
python scripts/81_compute_baselines.py
```

All scripts accept repository-relative defaults. Use `--data_dir` to point to a different leakage-safe processed dataset directory.

## Outputs

Experiment outputs are written under:

```text
outputs/mamba_final/
outputs/leakage_safe_rerun/
```

These output directories are ignored by Git.

## Repository Layout

```text
scLifeMamba/
  README.md
  requirements.txt
  environment.yml
  configs/
  docs/
  scripts/
    run_mamba_final_experiments.py
    run_final_audit_experiments.py
    81_compute_baselines.py
  src/
    data/
    dataset/
    models/
      torch_selective_ssm.py
      mamba_block.py
      scLifeMamba.py
    utils/
  tests/
```

## Limitations

- Current validated analysis uses one PBMC CITE-seq dataset.
- The reported Mamba backend is `TorchSelectiveSSM`, a native PyTorch implementation, not the official CUDA kernel.
- The fair deep comparison in the manuscript uses an 8k subset at horizon 1.
- Sequence order ablation indicates that at L=32 and h=1, per-cell feature expression carries most of the classification signal.

## Citation

```bibtex
@article{du2026sclifemamba,
  title={scLifeMamba: Trajectory-Aware Mamba-LSTM Framework for Multimodal Single-Cell State Modeling},
  author={Du, Wendong and Mao, Tengyue and Xiong, Wei and Chen, Lvyi and Liu, Cong},
  year={2026}
}
```

## License

MIT License.
