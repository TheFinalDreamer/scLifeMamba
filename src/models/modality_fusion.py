"""Three-modality state-aware dynamic fusion with missing modality support."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateAwareFusion(nn.Module):
    """Dynamic multi-modality fusion: RNA + ATAC + Protein with learned per-cell weights.

    alpha = softmax(MLP([z_rna, z_atac, z_protein]))
    z_fused = sum(alpha_i * z_i)

    Supports missing modalities via modality mask.
    """

    def __init__(self, rna_dim: int, atac_dim: int = None, protein_dim: int = None,
                 hidden_dim: int = 128, use_atac: bool = False, use_protein: bool = False):
        super().__init__()
        self.use_atac = use_atac
        self.use_protein = use_protein
        self.hidden_dim = hidden_dim

        # Project each modality to common hidden_dim
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        self.atac_proj = nn.Linear(atac_dim, hidden_dim) if use_atac and atac_dim else None
        self.protein_proj = nn.Linear(protein_dim, hidden_dim) if use_protein and protein_dim else None

        # Number of active modalities
        n_modalities = 1 + int(use_atac) + int(use_protein)

        # Gating network for dynamic weights
        if n_modalities > 1:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * n_modalities, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_modalities),
            )
        else:
            self.gate = None

    def forward(self, z_rna: torch.Tensor, z_atac: torch.Tensor = None,
                z_protein: torch.Tensor = None, modality_mask: torch.Tensor = None):
        """Args:
            z_rna: (B, rna_dim) — always required
            z_atac: (B, atac_dim) — optional
            z_protein: (B, protein_dim) — optional
            modality_mask: (B, n_modalities) — 1=present, 0=missing

        Returns:
            z_fused: (B, hidden_dim)
            modality_weights: (B, n_modalities)
        """
        h_rna = self.rna_proj(z_rna)
        components = [h_rna]
        index = 1

        if self.use_atac and z_atac is not None:
            components.append(self.atac_proj(z_atac))
        elif self.use_atac:
            components.append(torch.zeros_like(h_rna))
            index += 1

        if self.use_protein and z_protein is not None:
            components.append(self.protein_proj(z_protein))
        elif self.use_protein:
            components.append(torch.zeros_like(h_rna))

        if self.gate is not None:
            gate_input = torch.cat(components, dim=-1)
            alpha = F.softmax(self.gate(gate_input), dim=-1)
            if modality_mask is not None:
                alpha = alpha * modality_mask
                alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)
            z_fused = sum(alpha[:, i:i + 1] * comp for i, comp in enumerate(components))
        else:
            z_fused = components[0]
            alpha = torch.ones(z_rna.size(0), 1, device=z_rna.device)

        return z_fused, alpha


class ConcatFusion(nn.Module):
    """Simple concatenation fusion baseline (no dynamic weights)."""

    def __init__(self, rna_dim: int, atac_dim: int = None, protein_dim: int = None,
                 hidden_dim: int = 128, use_atac: bool = False, use_protein: bool = False):
        super().__init__()
        total_dim = rna_dim
        if use_atac and atac_dim:
            total_dim += atac_dim
        if use_protein and protein_dim:
            total_dim += protein_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim
        self.use_atac = use_atac
        self.use_protein = use_protein

    def forward(self, z_rna: torch.Tensor, z_atac: torch.Tensor = None,
                z_protein: torch.Tensor = None, modality_mask: torch.Tensor = None):
        parts = [z_rna]
        if self.use_atac and z_atac is not None:
            parts.append(z_atac)
        elif self.use_atac:
            parts.append(torch.zeros(z_rna.size(0), 0, device=z_rna.device))
        if self.use_protein and z_protein is not None:
            parts.append(z_protein)
        elif self.use_protein:
            parts.append(torch.zeros(z_rna.size(0), 0, device=z_rna.device))
        z = torch.cat(parts, dim=-1)
        z_fused = self.fusion(z)
        return z_fused, None
