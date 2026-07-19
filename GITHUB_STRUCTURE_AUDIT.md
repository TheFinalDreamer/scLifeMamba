# GITHUB_STRUCTURE_AUDIT

Date: 2026-07-19
Scope: `GITHUB_RELEASE/scLifeMamba`

## Initial Finding

The repository was clean before this release pass, but README commands referenced files that were not present in the release tree:

- `scripts/run_mamba_final_experiments.py`
- `scripts/run_final_audit_experiments.py`
- `scripts/81_compute_baselines.py`
- `src/models/torch_selective_ssm.py`

README also documented 15 epochs while the manuscript and supplementary material define the fair comparison as 10 epochs.

## Structure Added Or Synchronized

| Path | Purpose | Status |
|---|---|---|
| `scripts/run_mamba_final_experiments.py` | Main Mamba-LSTM experiment runner | PRESENT |
| `scripts/run_final_audit_experiments.py` | Fair backbone and sequence-order audit runner | PRESENT |
| `scripts/81_compute_baselines.py` | Leakage-safe classical baseline matrix | PRESENT |
| `src/models/torch_selective_ssm.py` | Native PyTorch selective SSM backend used in manuscript | PRESENT |
| `src/models/mamba_block.py` | Backend selector, now preferring native `mamba_ssm` if installed and otherwise `TorchSelectiveSSM` | PRESENT |
| `src/models/scLifeMamba.py` | Sequence-compatible model interface with `get_mamba_backend()` | PRESENT |
| `src/data/` | Data loading modules used by scripts | PRESENT |
| `src/dataset/` | Dataset namespace included for release structure compatibility | PRESENT |

## README Command Mapping

| README command | Target exists | Notes |
|---|---|---|
| `python scripts/run_mamba_final_experiments.py` | YES | Uses repository-relative default data path and 10 epochs in README |
| `python scripts/run_final_audit_experiments.py --exp fair --epochs 10` | YES | Matches manuscript fair comparison epoch count |
| `python scripts/run_final_audit_experiments.py --exp order --epochs 10` | YES | Matches manuscript sequence-order ablation |
| `python scripts/81_compute_baselines.py` | YES | Runs leakage-safe classical baseline matrix |

## Exclusions

The release repository does not track:

- raw data
- `.h5ad` or `.h5` files
- processed `.npy`, `.npz`, or parquet data
- checkpoints
- logs
- experiment outputs
- temporary build artifacts

These exclusions are enforced by `.gitignore`.

## Validation

Executed:

```bash
python -m py_compile scripts/run_mamba_final_experiments.py scripts/run_final_audit_experiments.py scripts/81_compute_baselines.py src/models/torch_selective_ssm.py src/models/mamba_block.py src/data/sequence_dataloader.py
```

Executed model smoke test:

```text
logits (2, 3)
backend torch_selective_ssm
```

README forbidden-term audit:

```text
SOTA: 0
best: 0
superior: 0
outperform: 0
```

## Verdict

README command paths now map to real repository files. The GitHub release structure is aligned with the manuscript code availability statement.
