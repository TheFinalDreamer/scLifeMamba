"""RNA and Protein encoders with MLP + LayerNorm + Dropout."""

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: list, dropout: float) -> nn.Sequential:
    """Build an MLP with LayerNorm and Dropout between layers."""
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = h_dim
    return nn.Sequential(*layers)


class RNAEncoder(nn.Module):
    """Encode RNA expression features into latent representation.

    Architecture: MLP + LayerNorm + ReLU + Dropout at each hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()
        self.mlp = _build_mlp(input_dim, hidden_dims, dropout)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, rna_dim) RNA expression features

        Returns:
            z_rna: (batch_size, output_dim) encoded RNA representation
        """
        return self.mlp(x)


class ProteinEncoder(nn.Module):
    """Encode Protein expression features into latent representation.

    Architecture: MLP + LayerNorm + ReLU + Dropout at each hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()
        self.mlp = _build_mlp(input_dim, hidden_dims, dropout)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, protein_dim) Protein expression features

        Returns:
            z_protein: (batch_size, output_dim) encoded protein representation
        """
        return self.mlp(x)
