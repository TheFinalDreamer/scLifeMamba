"""State-Aware Dynamic Modality Fusion module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateAwareFusion(nn.Module):
    """Dynamically fuse RNA and Protein representations based on cell state context.

    Computes modality weights via a gating MLP:
        alpha = softmax(MLP([z_rna, z_protein]))
        z_fused = alpha_rna * z_rna + alpha_protein * z_protein
    """

    def __init__(self, rna_dim: int, protein_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.rna_dim = rna_dim
        self.protein_dim = protein_dim

        # Ensure both modalities project to same dimension for fusion
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)

        # Gating network to compute dynamic weights
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.output_dim = hidden_dim

    def forward(self, z_rna: torch.Tensor, z_protein: torch.Tensor):
        """
        Args:
            z_rna: (batch_size, rna_dim)
            z_protein: (batch_size, protein_dim)

        Returns:
            z_fused: (batch_size, hidden_dim)
            modality_weights: (batch_size, 2) [alpha_rna, alpha_protein]
        """
        # Project to common space
        h_rna = self.rna_proj(z_rna)
        h_protein = self.protein_proj(z_protein)

        # Compute gate logits and softmax weights
        gate_input = torch.cat([h_rna, h_protein], dim=-1)
        alpha = F.softmax(self.gate(gate_input), dim=-1)  # (batch, 2)

        # Weighted fusion
        z_fused = alpha[:, 0:1] * h_rna + alpha[:, 1:2] * h_protein

        return z_fused, alpha


class SimpleFusion(nn.Module):
    """Simple concatenation-based fusion (no dynamic weights), used as baseline."""

    def __init__(self, rna_dim: int, protein_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(rna_dim + protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, z_rna: torch.Tensor, z_protein: torch.Tensor):
        z = torch.cat([z_rna, z_protein], dim=-1)
        z_fused = self.fusion(z)
        return z_fused, None
