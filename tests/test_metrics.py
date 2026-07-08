"""Test evaluation metrics."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pytest

from src.evaluation.metrics import compute_classification_metrics, compute_pseudotime_metrics
from src.evaluation.trajectory_metrics import compute_trajectory_metrics
from src.evaluation.efficiency import compute_efficiency_metrics


class TestClassificationMetrics:
    """Test classification metric computation."""

    def test_perfect_prediction(self):
        logits = torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 0.0],
        ])
        labels = torch.tensor([0, 1, 2, 0])
        metrics = compute_classification_metrics(logits, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_random_prediction(self):
        torch.manual_seed(42)
        logits = torch.randn(100, 5)
        labels = torch.randint(0, 5, (100,))
        metrics = compute_classification_metrics(logits, labels)
        assert 0 <= metrics["accuracy"] <= 1
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics

    def test_single_class(self):
        logits = torch.randn(10, 3)
        labels = torch.zeros(10, dtype=torch.long)
        metrics = compute_classification_metrics(logits, labels)
        assert metrics["accuracy"] >= 0


class TestPseudotimeMetrics:
    """Test pseudotime metric computation."""

    def test_perfect_prediction(self):
        pred = torch.linspace(0, 1, 50)
        true = torch.linspace(0, 1, 50)
        metrics = compute_pseudotime_metrics(pred, true)
        assert metrics["mae"] < 1e-4
        assert metrics["mse"] < 1e-4
        assert metrics["spearman"] > 0.99

    def test_random_prediction(self):
        torch.manual_seed(42)
        pred = torch.rand(100)
        true = torch.rand(100)
        metrics = compute_pseudotime_metrics(pred, true)
        assert "spearman" in metrics
        assert "mae" in metrics
        assert "mse" in metrics

    def test_constant_prediction(self):
        pred = torch.ones(50) * 0.5
        true = torch.linspace(0, 1, 50)
        metrics = compute_pseudotime_metrics(pred, true)
        assert metrics["mae"] >= 0


class TestTrajectoryMetrics:
    """Test trajectory structure metrics."""

    def test_basic_metrics(self):
        np.random.seed(42)
        embeddings = np.random.randn(100, 32)
        labels = np.random.randint(0, 5, 100)
        pseudotime = np.linspace(0, 1, 100)

        metrics = compute_trajectory_metrics(embeddings, labels, pseudotime)
        assert "silhouette" in metrics
        assert "ari" in metrics
        assert "nmi" in metrics
        assert "neighborhood_preservation" in metrics

    def test_single_class(self):
        embeddings = np.random.randn(50, 10)
        labels = np.zeros(50, dtype=np.int64)
        metrics = compute_trajectory_metrics(embeddings, labels)
        # All should be None or valid with single class
        assert "silhouette" in metrics


class TestEfficiencyMetrics:
    """Test efficiency metric computation."""

    def test_param_count(self):
        import torch.nn as nn
        model = nn.Linear(10, 5)
        metrics = compute_efficiency_metrics(model)
        assert metrics["n_params"] == 55  # 10*5 + 5 bias
        assert "inference_time_ms" in metrics
