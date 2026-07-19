# Mamba Backend

The manuscript reports the `TorchSelectiveSSM` backend.

## TorchSelectiveSSM

File:

```text
src/models/torch_selective_ssm.py
```

This is a native PyTorch implementation of selective state-space modeling. It includes:

- input-dependent delta projection
- input-dependent B and C projections
- stable learned A parameterization
- selective recurrence implemented in PyTorch
- CUDA support through standard PyTorch tensors

This repository provides the implementation used in the manuscript. It is not a wrapper around the Linux-only `mamba_ssm` package.

## Native mamba_ssm

The native CUDA package may be useful for future speed optimization on Linux systems, but it is not required here and is not listed in `requirements.txt`.

## Runtime Selection

`src/models/mamba_block.py` uses native `mamba_ssm` only if it is already installed. Otherwise it loads `TorchSelectiveSSMBlock`. A Conv1d/GRU diagnostic fallback remains available only when neither selective backend can be imported.
