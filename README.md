# scLifeMamba: Trajectory-Aware Mamba-LSTM Framework for Multimodal Single-Cell State Modeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A trajectory-aware Mamba-LSTM architecture for multimodal single-cell state modeling under donor-held-out leakage-controlled evaluation.

---

## Architecture

scLifeMamba combines five components for pseudotime-ordered multimodal sequence modeling:

```
RNA (1000 HVGs) ──→ RNAEncoder ──→ z_rna ──┐
                                              ├──→ DynamicFusion ──→ MambaBlock ──→ LSTM ──→ Prediction
Protein (228 ADTs) → ProteinEncoder → z_prot ┘        ↑                      ↑          ↑
                                                  learned gate        selective SSM    local recurrence
```

- **RNA/Protein Encoders**: MLP-based per-cell feature extraction
- **Dynamic Modality Fusion**: Learned per-position gating (softmax over RNA+Protein weights)
- **Mamba Block**: Selective state-space model (Gu & Dao, 2023) for long-range dependency capture
  - Implemented as `TorchSelectiveSSM`: pure PyTorch selective SSM with CUDA validation
  - Full algorithm: input-dependent Δ, B, C projections, HiPPO-init A, parallel associative scan
  - Cross-platform alternative to Linux-only `mamba_ssm` CUDA kernels
- **LSTM**: 2-layer unidirectional LSTM for local transition refinement
- **γ-Fusion**: Learned weighted combination of Mamba + LSTM outputs

---

## Key Results

Evaluated on 161,764 PBMC CITE-seq cells under donor-held-out protocol (5 train / 2 val / 1 test donors):

| Model | Macro F1 | Accuracy | Params |
|-------|----------|----------|--------|
| LR (Classical) | 0.8796 ± 0.0001 | 0.9044 ± 0.0001 | -- |
| MLP (mean-pool) | 0.9340* | 0.9464* | 763,783 |
| LSTM-only | 0.9254** | 0.9424** | 1,044,743 |
| Mamba-only | 0.9054** | 0.9304** | 897,287 |
| **Mamba-LSTM** | **0.9268*** | **0.9409*** | **1,161,480** |

\* Full data (161k sequences), single seed  
\** Subset result (10k sequences, 5 epochs)  
Mamba backend: TorchSelectiveSSM (PyTorch). Classical baselines: 3 seeds × 4 horizons.

All deep models substantially exceed classical baselines. MLP with mean-pooling achieves the highest F1 on this task, suggesting that for 32-cell windows at horizon 1, per-cell feature expression is highly informative. The Mamba-LSTM architecture provides competitive sequence modeling performance with the architectural advantage of linear-time sequence processing for longer contexts.

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.5+ with CUDA
- RTX 4080 or equivalent (16GB+ VRAM recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/TheFinalDreamer/scLifeMamba.git
cd scLifeMamba

# Install dependencies
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install scanpy scikit-learn pandas numpy pyarrow matplotlib pyyaml anndata

# Optional: native mamba_ssm (Linux only, not required)
# pip install mamba-ssm causal-conv1d
```

**Note:** scLifeMamba uses `TorchSelectiveSSM` — a pure PyTorch implementation of the Mamba selective SSM algorithm. This works on any platform (Windows/Linux/Mac with CUDA) without requiring the Linux-specific `mamba_ssm` package. The algorithm is mathematically equivalent; only CUDA kernel optimization differs.

---

## Dataset Preparation

The PBMC CITE-seq dataset is from Seurat v4 (Hao et al., 2021, Cell). Download the h5seurat file:

```bash
# Download from Seurat reference
wget https://atlas.fredhutch.org/data/nygc/multimodal/pbmc_multimodal.h5seurat
```

Or use the preprocessed leakage-safe data:

```bash
# Preprocessed data is at data/processed/leakage_safe_v1/
# Contains:
#   - rna_hvg_all.npy (161764 × 1000)
#   - protein_norm_all.npy (161764 × 228)
#   - reference_sequence_manifest.parquet (643,384 sequences)
#   - split_assignments.csv
```

---

## Training

### Quick Start (Subset)

```bash
cd code
python scripts/run_mamba_final_experiments.py \
    --exp backbone \
    --model mamba_lstm \
    --horizon 1 \
    --seed 42 \
    --epochs 15 \
    --batch_size 64
```

### Fair Backbone Comparison

```bash
cd code
python scripts/run_final_audit_experiments.py \
    --exp fair \
    --epochs 15 \
    --lr 1e-3
```

### Sequence Order Ablation

```bash
cd code
python scripts/run_final_audit_experiments.py \
    --exp order \
    --epochs 15 \
    --lr 1e-3
```

---

## Evaluation

Results are saved to `outputs/mamba_final/audit/` with the structure:

```
outputs/mamba_final/audit/
├── exp1_fair/
│   ├── mlp/seed42/h1/
│   │   ├── metrics.json
│   │   ├── config.json
│   │   └── predictions.csv
│   ├── lstm/...
│   ├── mamba/...
│   └── mamba_lstm/...
│   └── aggregate_results.json
└── exp2_order/
    ├── original/...
    ├── random/...
    └── aggregate_results.json
```

Each `metrics.json` contains: `test_macro_f1`, `test_accuracy`, `test_balanced_accuracy`, `test_per_class_f1`, `confusion_matrix`, training history.

---

## Reproduce Paper Results

To reproduce the main results from the manuscript:

```bash
# 1. Fair backbone comparison (all models, 3 seeds, 15 epochs)
python code/scripts/run_final_audit_experiments.py --exp fair --epochs 15

# 2. Sequence order ablation (original vs shuffled)
python code/scripts/run_final_audit_experiments.py --exp order --epochs 15

# 3. Classical baselines
python code/scripts/81_compute_baselines.py
```

---

## Project Structure

```
├── code/
│   ├── src/
│   │   ├── models/
│   │   │   ├── scLifeMamba.py          # Main model assembly
│   │   │   ├── mamba_lstm.py           # Mamba-LSTM sequence encoder
│   │   │   ├── mamba_block.py          # Mamba block (3-tier backend)
│   │   │   ├── torch_selective_ssm.py  # Pure PyTorch selective SSM
│   │   │   ├── encoders.py             # RNA/Protein MLP encoders
│   │   │   ├── fusion.py               # Dynamic/Simple modality fusion
│   │   │   └── heads.py                # Classification/Pseudotime/Embedding heads
│   │   ├── data/
│   │   │   ├── sequence_dataloader.py   # Fast parquet-based sequence loader
│   │   │   ├── dataset.py              # Single-cell dataset
│   │   │   └── preprocessing.py        # Leakage-safe preprocessing
│   │   ├── training/
│   │   │   ├── trainer.py
│   │   │   └── early_stopping.py
│   │   └── evaluation/
│   │       └── metrics.py
│   └── scripts/
│       ├── run_mamba_final_experiments.py   # Main experiment runner
│       ├── run_final_audit_experiments.py   # Fair audit experiments
│       ├── preprocess_dataset.py            # Data preprocessing
│       └── 81_compute_baselines.py          # Classical baselines
├── data/processed/leakage_safe_v1/          # Preprocessed data
├── outputs/                                  # Experiment outputs
│   ├── leakage_safe_rerun/                  # Classical baseline results
│   └── mamba_final/                         # Deep learning results
├── manuscript/bioinformatics_submission_draft_v1/  # Manuscript
└── FINAL_SUBMISSION_AUDIT/                  # Submission audit reports
```

---

## Limitations

- Single PBMC CITE-seq dataset; cross-dataset validation needed
- Mamba SSM uses pure PyTorch implementation (TorchSelectiveSSM), not native `mamba_ssm` CUDA kernel
- Lifecycle labels are pseudotime-derived operational bins
- Training comparison across models used different epoch budgets in early runs (being corrected in ongoing multi-seed experiments)
- No comparison with large-scale models (totalVI, scGPT, MultiVI) — planned for future work

---

## Citation

If you use scLifeMamba in your research, please cite:

```bibtex
@article{du2026sclifemamba,
  title={scLifeMamba: Trajectory-Aware Mamba-LSTM Framework for Multimodal Single-Cell State Modeling},
  author={Du, Wendong and Mao, Tengyue and Xiong, Wei and Chen, Lvyi and Liu, Cong},
  journal={Bioinformatics},
  year={2026},
  note={Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Wendong Du — School of Computer Science and Artificial Intelligence, South-Central Minzu University  
Corresponding author: Tengyue Mao — 3038807@mail.scuec.edu.cn
