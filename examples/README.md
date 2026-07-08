# Examples

## Minimal smoke test

```bash
python examples/minimal_smoke_test.py
```

Verifies that all core modules import and run correctly using synthetic data. No GPU or real data required. Expected output: `5/5 tests passed`.

## What the smoke test covers

1. MambaBlock instantiation and forward pass (fallback backend)
2. MambaLSTMEncoder with gated Mamba+LSTM fusion
3. LagAwareDynamicFusion with pseudotime and horizon embeddings
4. scMultiLifeMamba full model with RNA+Protein modalities
5. Direction label reconstruction logic verification

## Example config

See `example_config.yaml` for a minimal lifecycle prediction configuration.
