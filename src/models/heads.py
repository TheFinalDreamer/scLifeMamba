"""Prediction heads for classification, pseudotime regression, and embedding."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP head for cell state classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        return self.net(x)


class PseudotimeHead(nn.Module):
    """MLP head for pseudotime regression. Outputs a scalar in [0, 1]."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            pseudotime_pred: (batch_size, 1)
        """
        return self.net(x)


class EmbeddingHead(nn.Module):
    """Projection head for low-dimensional embedding (UMAP/trajectory analysis)."""

    def __init__(self, input_dim: int, embedding_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            embedding: (batch_size, embedding_dim)
        """
        return self.net(x)
