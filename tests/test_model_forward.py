"""Test scLifeMamba model forward pass with synthetic data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.models.scLifeMamba import scLifeMamba
from src.models.baselines import (
    MLPBaseline,
    LSTMBaseline,
    TransformerBaseline,
    MambaOnlyBaseline,
)


@pytest.fixture
def sample_batch():
    batch_size = 32
    rna_dim = 100
    protein_dim = 20
    return {
        "x_rna": torch.randn(batch_size, rna_dim),
        "x_protein": torch.randn(batch_size, protein_dim),
    }


@pytest.fixture
def model_kwargs():
    return {
        "rna_dim": 100,
        "protein_dim": 20,
        "num_classes": 5,
        "rna_hidden_dims": [64, 32],
        "protein_hidden_dims": [32, 16],
        "hidden_dim": 64,
        "embedding_dim": 32,
        "dropout": 0.1,
    }


class TestscLifeMamba:
    """Test the main scLifeMamba model."""

    def test_forward_output_keys(self, sample_batch, model_kwargs):
        model = scLifeMamba(**model_kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])

        assert "logits" in out
        assert "pseudotime_pred" in out
        assert "embedding" in out
        assert "modality_weights" in out

    def test_output_shapes(self, sample_batch, model_kwargs):
        batch_size = sample_batch["x_rna"].shape[0]
        model = scLifeMamba(**model_kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])

        assert out["logits"].shape == (batch_size, model_kwargs["num_classes"])
        assert out["pseudotime_pred"].shape == (batch_size, 1)
        assert out["embedding"].shape == (batch_size, model_kwargs["embedding_dim"])
        assert out["modality_weights"].shape == (batch_size, 2)

    def test_modality_weights_sum_to_one(self, sample_batch, model_kwargs):
        model = scLifeMamba(**model_kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])

        weights = out["modality_weights"]
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_gradient_flow(self, sample_batch, model_kwargs):
        model = scLifeMamba(**model_kwargs)
        model.train()
        out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        # Sum all outputs to ensure gradients flow through all heads
        loss = out["logits"].sum() + out["pseudotime_pred"].sum() + out["embedding"].sum()
        loss.backward()

        # LSTM hidden-to-hidden weights may have zero grad with seq_len=1,
        # which is expected behavior. Skip those parameters.
        skip_patterns = ["lstm.weight_hh", "lstm.bias_hh"]
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                if not any(p in name for p in skip_patterns):
                    assert not torch.allclose(
                        param.grad, torch.zeros_like(param.grad)
                    ), f"Zero gradient for {name}"

    def test_without_mamba(self, sample_batch, model_kwargs):
        kwargs = {**model_kwargs, "use_mamba": False}
        model = scLifeMamba(**kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        assert out["logits"].shape[1] == model_kwargs["num_classes"]

    def test_without_lstm(self, sample_batch, model_kwargs):
        kwargs = {**model_kwargs, "use_lstm": False}
        model = scLifeMamba(**kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        assert out["logits"].shape[1] == model_kwargs["num_classes"]

    def test_without_fusion(self, sample_batch, model_kwargs):
        kwargs = {**model_kwargs, "use_dynamic_fusion": False}
        model = scLifeMamba(**kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        assert out["logits"].shape[1] == model_kwargs["num_classes"]
        # Simple fusion produces None modality_weights
        assert out["modality_weights"] is None

    def test_without_both_mamba_and_lstm(self, sample_batch, model_kwargs):
        kwargs = {**model_kwargs, "use_mamba": False, "use_lstm": False}
        model = scLifeMamba(**kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        assert out["logits"].shape[1] == model_kwargs["num_classes"]


class TestBaselines:
    """Test baseline model forward passes."""

    def test_mlp_baseline(self, sample_batch, model_kwargs):
        model = MLPBaseline(**model_kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        assert "logits" in out
        assert out["modality_weights"] is None

    def test_lstm_baseline(self, sample_batch, model_kwargs):
        model = LSTMBaseline(**model_kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        assert out["logits"].shape[1] == model_kwargs["num_classes"]

    def test_transformer_baseline(self, sample_batch, model_kwargs):
        kwargs = {**model_kwargs, "transformer_num_heads": 2}  # must divide hidden dim
        model = TransformerBaseline(**kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        assert out["logits"].shape[1] == model_kwargs["num_classes"]

    def test_mamba_only_baseline(self, sample_batch, model_kwargs):
        model = MambaOnlyBaseline(**model_kwargs)
        model.eval()
        with torch.no_grad():
            out = model(sample_batch["x_rna"], sample_batch["x_protein"])
        assert out["logits"].shape[1] == model_kwargs["num_classes"]
