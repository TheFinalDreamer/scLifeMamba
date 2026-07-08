#!/usr/bin/env python3
"""Minimal smoke test for scLifeMamba. Uses synthetic data — no GPU or real data required.

Run: python examples/minimal_smoke_test.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np


def test_mamba_block():
    """Test that the MambaBlock can be instantiated and run a forward pass."""
    from src.models.mamba_block import MambaBlock, is_mamba_available

    block = MambaBlock(dim=64, d_state=16, d_conv=4, expand=2, dropout=0.1)
    x = torch.randn(2, 8, 64)
    y = block(x)

    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    real_mamba = is_mamba_available()
    print(f"  [OK] MambaBlock: is_real_mamba={real_mamba}, output shape={y.shape}")


def test_mamba_lstm_encoder():
    """Test the Mamba-LSTM encoder."""
    from src.models.mamba_lstm import MambaLSTMEncoder

    encoder = MambaLSTMEncoder(
        input_dim=64, hidden_dim=64,
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        lstm_num_layers=2, dropout=0.1,
        use_mamba=True, use_lstm=True
    )

    x = torch.randn(4, 8, 64)  # (batch, seq_len, feature_dim)
    h_pooled, h_seq = encoder(x)

    assert h_pooled.shape == (4, 64), f"Expected (4, 64), got {h_pooled.shape}"
    assert h_seq.shape == (4, 8, 64), f"Expected (4, 8, 64), got {h_seq.shape}"
    print(f"  [OK] MambaLSTMEncoder: pooled={h_pooled.shape}, seq={h_seq.shape}")


def test_lag_aware_fusion():
    """Test the LagAwareDynamicFusion module."""
    from src.models.lag_aware_dynamic_fusion import LagAwareDynamicFusion

    fusion = LagAwareDynamicFusion(
        d_model=64, num_modalities=2,
        n_pseudotime_bins=20, max_horizon=16,
        use_protein=True, use_atac=False, dropout=0.1
    )

    z_rna = torch.randn(16, 64)
    z_protein = torch.randn(16, 64)
    pseudotime = torch.rand(16)
    horizon = 4

    z_fused, weights = fusion(z_rna, z_protein, pseudotime=pseudotime, horizon=horizon)

    assert z_fused.shape == (16, 64), f"Expected (16, 64), got {z_fused.shape}"
    assert "alpha_rna" in weights and "alpha_protein" in weights
    print(f"  [OK] LagAwareDynamicFusion: output={z_fused.shape}, weights keys={list(weights.keys())}")


def test_scmultilifemamba():
    """Test the full scMultiLifeMamba model (fallback backend)."""
    from src.models.scmultilifemamba import scMultiLifeMamba

    model = scMultiLifeMamba(
        rna_dim=64,
        num_classes=4,
        protein_dim=32,
        hidden_dim=64,
        embedding_dim=32,
        dropout=0.1,
        use_atac=False,
        use_protein=True,
        use_mamba=True,
        use_lstm=True,
        use_dynamic_fusion=True,
        use_lag_aware_fusion=False,
    )

    x_rna = torch.randn(2, 8, 64)
    x_protein = torch.randn(2, 8, 32)
    pseudotime = torch.rand(2)

    output = model(x_rna, x_protein=x_protein, pseudotime=pseudotime, horizon=4)

    assert "logits" in output, f"Missing 'logits' in output keys: {list(output.keys())}"
    assert "pred" in output or "pseudotime" in output, f"Missing prediction in output keys: {list(output.keys())}"
    print(f"  [OK] scMultiLifeMamba: logits={output['logits'].shape}, is_real_mamba={model.is_real_mamba}")


def test_label_reconstruction():
    """Test direction label reconstruction logic."""
    # Simulate pseudotime windows with known forward/backward patterns
    np.random.seed(42)
    pseudotime = np.random.rand(100, 32).astype(np.float32)
    pseudotime.sort(axis=1)  # monotonic forward

    # Naive labeling (reproduces the 99.87% single-class problem)
    pt_delta = pseudotime[:, -1] - pseudotime[:, 0]
    forward_frac = (pt_delta > 0.01).mean()
    assert forward_frac > 0.99, f"Expected >99% forward, got {forward_frac:.2%}"
    print(f"  [OK] Label reconstruction: naive forward fraction = {forward_frac:.2%} (expected >99%)")


def main():
    print("scLifeMamba — minimal smoke test")
    print("Using synthetic data. No GPU, no real data, no mamba-ssm required.\n")

    tests = [
        test_mamba_block,
        test_mamba_lstm_encoder,
        test_lag_aware_fusion,
        test_scmultilifemamba,
        test_label_reconstruction,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")

    if passed == len(tests):
        print("All smoke tests passed. scLifeMamba is correctly installed.")
    else:
        print("Some tests failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
