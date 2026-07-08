# Mamba backend

scLifeMamba supports two backends for the Mamba state-space model block.

## Native backend (production)

Requires `mamba-ssm` and `causal-conv1d` packages:

```bash
pip install mamba-ssm causal-conv1d
```

These packages require:
- Linux with CUDA toolkit
- NVIDIA GPU with compute capability >= 7.0
- PyTorch with CUDA support

The native backend implements the selective state-space scanning described in Gu & Dao (2023). It provides linear-time sequence modeling and is used for full-scale training experiments.

## Fallback backend (development)

Automatically used when `mamba-ssm` is not available. Implements:

- Depthwise Conv1d with symmetric padding
- SiLU activation
- Gated projection (sigmoid gate)
- GRU for sequence recurrence
- Residual connection with LayerNorm

The fallback is functionally similar but does not implement selective scanning. It is suitable for:

- Code development and debugging
- CPU-only environments (Windows, macOS)
- Smoke tests and unit tests
- Modality contribution comparisons (same backend for all configurations)

## Switching backends

Backend selection is automatic:

```python
from src.models.mamba_block import MambaBlock, is_mamba_available

block = MambaBlock(dim=128)
print(block.is_real_mamba)  # True if mamba-ssm is installed
```

No manual configuration is needed.

## Reporting results

When publishing results, specify which backend was used:

- "Native Mamba" = `mamba-ssm` on Linux GPU
- "Fallback" = Conv1d+GRU on any platform

Modality comparison results (protein-only vs RNA-only vs fusion) are valid under either backend when all configurations use the same backend.
