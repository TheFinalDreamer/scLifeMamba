# Reproducibility

## Environment

- Python 3.10+
- PyTorch 2.0+
- Scanpy 1.9+
- See `requirements.txt` for full dependency list

## Backend selection

| Environment | Backend | `mamba-ssm` required |
|-------------|---------|---------------------|
| Linux + CUDA | Native Mamba | Yes |
| Windows / CPU | Fallback | No |

The fallback backend uses a Conv1d + GRU approximation. When comparing modality contributions (protein-only vs RNA-only vs fusion), the same backend is used for all configurations, ensuring valid within-study comparisons.

## Random seeds

All experiments use fixed random seeds (42, 43, 44) for PyTorch, NumPy, and Python random. Seeds are set at the start of each experiment script.

## Data split

Train/validation/test split: 70% / 15% / 15%, random permutation with seed 42. The same split indices are used across all model configurations for a given task.

## Experiment tracking

Each run produces:

```
outputs/<task>/<timestamp>/
├── run_status.json    # Completion status, model name, hyperparameters
├── metrics.json       # Classification/regression metrics
└── config.json        # Full runtime configuration
```

## Expected variation

- Fallback vs native Mamba: Minor metric differences expected. Modality comparison conclusions are consistent across backends.
- Seed variation: Small variance across 3 seeds; aggregate results are reported as mean across seeds.
- Hardware: GPU vs CPU does not affect model output for the same seed and backend.
