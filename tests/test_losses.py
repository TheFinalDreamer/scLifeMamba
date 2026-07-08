"""Test loss functions with synthetic data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.utils.config import Config
from src.losses.multitask_loss import MultiTaskLoss
from src.losses.trajectory_loss import TrajectoryLoss
from src.losses.contrastive_loss import ModalityContrastiveLoss


@pytest.fixture
def config():
    return Config({
        "loss": {
            "use_cls_loss": True,
            "use_ptime_loss": True,
            "use_traj_loss": True,
            "use_contrast_loss": True,
            "lambda_ptime": 1.0,
            "lambda_traj": 0.1,
            "lambda_contrast": 0.1,
            "ptime_loss_type": "mse",
            "contrast_temperature": 0.07,
        }
    })


@pytest.fixture
def batch():
    batch_size = 32
    return {
        "label": torch.randint(0, 5, (batch_size,)),
        "pseudotime": torch.rand(batch_size),
    }


@pytest.fixture
def predictions():
    batch_size = 32
    return {
        "logits": torch.randn(batch_size, 5),
        "pseudotime_pred": torch.rand(batch_size, 1),
        "embedding": torch.randn(batch_size, 64),
    }


class TestMultiTaskLoss:
    """Test the combined multi-task loss."""

    def test_forward_all_components(self, config, predictions, batch):
        loss_fn = MultiTaskLoss(config)
        # Use a full adjacency matrix (not identity) so trajectory loss is non-zero
        adjacency = torch.rand(32, 32) * 0.3
        adjacency = (adjacency + adjacency.T) / 2
        adjacency.fill_diagonal_(0)
        z_rna = torch.randn(32, 32)
        z_protein = torch.randn(32, 16)

        losses_dict, total_loss = loss_fn(
            predictions, batch, adjacency=adjacency,
            z_rna=z_rna, z_protein=z_protein,
        )

        assert "loss" in losses_dict
        assert losses_dict["loss"] > 0
        assert losses_dict["loss_cls"] > 0
        assert losses_dict["loss_ptime"] > 0
        assert losses_dict["loss_traj"] > 0
        assert losses_dict["loss_contrast"] > 0
        assert isinstance(total_loss, torch.Tensor)

    def test_backward(self, config, predictions, batch):
        loss_fn = MultiTaskLoss(config)
        adjacency = torch.eye(32) * 0.5
        z_rna = torch.randn(32, 32)
        z_protein = torch.randn(32, 16)

        # Make predictions require grad
        for k in predictions:
            predictions[k] = predictions[k].clone().requires_grad_(True)

        _, total_loss = loss_fn(
            predictions, batch, adjacency=adjacency,
            z_rna=z_rna, z_protein=z_protein,
        )
        total_loss.backward()

        for k in predictions:
            assert predictions[k].grad is not None, f"No gradient for predictions['{k}']"

    def test_skip_missing_targets(self, config, predictions):
        loss_fn = MultiTaskLoss(config)
        batch_without_labels = {"label": None, "pseudotime": None}

        losses_dict, total_loss = loss_fn(
            predictions, batch_without_labels, adjacency=None,
            z_rna=None, z_protein=None,
        )

        assert losses_dict["loss"] == 0.0
        assert losses_dict["loss_cls"] == 0.0
        assert losses_dict["loss_ptime"] == 0.0
        assert losses_dict["loss_traj"] == 0.0
        assert losses_dict["loss_contrast"] == 0.0

    def test_disable_loss_components(self, predictions, batch):
        cfg = Config({
            "loss": {
                "use_cls_loss": False,
                "use_ptime_loss": False,
                "use_traj_loss": False,
                "use_contrast_loss": False,
                "lambda_ptime": 1.0,
                "lambda_traj": 0.1,
                "lambda_contrast": 0.1,
            }
        })
        loss_fn = MultiTaskLoss(cfg)
        adjacency = torch.eye(32) * 0.5
        z_rna = torch.randn(32, 32)
        z_protein = torch.randn(32, 16)

        losses_dict, total_loss = loss_fn(
            predictions, batch, adjacency=adjacency,
            z_rna=z_rna, z_protein=z_protein,
        )

        assert losses_dict["loss"] == 0.0


class TestTrajectoryLoss:
    """Test trajectory structure loss."""

    def test_forward(self):
        loss_fn = TrajectoryLoss()
        embeddings = torch.randn(10, 64)
        adjacency = torch.eye(10) * 0.5

        loss = loss_fn(embeddings, adjacency)
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0

    def test_none_adjacency(self):
        loss_fn = TrajectoryLoss()
        embeddings = torch.randn(10, 64)
        loss = loss_fn(embeddings, None)
        assert loss.item() == 0.0


class TestContrastiveLoss:
    """Test modality contrastive loss."""

    def test_forward(self):
        loss_fn = ModalityContrastiveLoss(temperature=0.07)
        z_rna = torch.randn(32, 64)
        z_protein = torch.randn(32, 64)

        loss = loss_fn(z_rna, z_protein)
        assert isinstance(loss, torch.Tensor)
        assert loss > 0

    def test_single_sample(self):
        loss_fn = ModalityContrastiveLoss()
        z_rna = torch.randn(1, 64)
        z_protein = torch.randn(1, 64)

        loss = loss_fn(z_rna, z_protein)
        assert loss.item() == 0.0
