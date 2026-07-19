"""
scLifeMamba: Mamba-LSTM Multi-modal Fusion Model for Single-Cell Analysis.

Architecture (RNA+Protein, sequence mode):
    x_rna (B, L, rna_dim) ──→ RNAEncoder (per-position) ──→ z_rna (B, L, 256)
    x_prot (B, L, prot_dim) ──→ ProteinEncoder (per-position) ──→ z_prot (B, L, 64)
                                        │
                        StateAwareFusion (per-position) ──→ z_fused (B, L, hidden_dim)
                                        │
                        MambaLSTMEncoder ──→ h_pooled (B, hidden_dim)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
            ClassificationHead   PseudotimeHead      EmbeddingHead

Supports both sequence mode (3D input) and single-cell mode (2D input, backward compat).
"""

import torch
import torch.nn as nn

from .encoders import RNAEncoder, ProteinEncoder
from .fusion import StateAwareFusion, SimpleFusion
from .mamba_lstm import MambaLSTMEncoder
from .heads import ClassificationHead, PseudotimeHead, EmbeddingHead


class scLifeMamba(nn.Module):
    """Mamba-LSTM multi-modal fusion model for single-cell state modeling.

    Supports:
      - RNA+Protein (sequence mode):  x_rna (B,L,D_rna), x_protein (B,L,D_prot)
      - RNA+Protein (single mode):    x_rna (B,D_rna), x_protein (B,D_prot)  [backward compat]
      - RNA-only (sequence mode):     x_rna (B,L,D_rna), x_protein=None
      - RNA-only (single mode):       x_rna (B,D_rna), x_protein=None

    Outputs (dict):
        - logits: classification logits (B, num_classes)
        - pseudotime_pred: predicted pseudotime (B, 1)
        - embedding: latent embedding (B, embedding_dim)
        - modality_weights: (B, 2) or None if use_protein=False
    """

    def __init__(
        self,
        rna_dim: int,
        protein_dim: int,
        num_classes: int,
        rna_hidden_dims: list = None,
        protein_hidden_dims: list = None,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.2,
        use_mamba: bool = True,
        use_lstm: bool = True,
        use_dynamic_fusion: bool = True,
        use_protein: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        lstm_num_layers: int = 2,
    ):
        super().__init__()

        if rna_hidden_dims is None:
            rna_hidden_dims = [512, 256]
        if protein_hidden_dims is None:
            protein_hidden_dims = [64]

        self.use_mamba = use_mamba
        self.use_lstm = use_lstm
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_protein = use_protein
        self.hidden_dim = hidden_dim

        # RNA encoder (always used)
        self.rna_encoder = RNAEncoder(rna_dim, rna_hidden_dims, dropout)
        rna_out_dim = self.rna_encoder.output_dim

        # Protein encoder (only if use_protein)
        if use_protein:
            self.protein_encoder = ProteinEncoder(protein_dim, protein_hidden_dims, dropout)
            protein_out_dim = self.protein_encoder.output_dim

            # Fusion
            if use_dynamic_fusion:
                self.fusion = StateAwareFusion(rna_out_dim, protein_out_dim, hidden_dim)
            else:
                self.fusion = SimpleFusion(rna_out_dim, protein_out_dim, hidden_dim)

            fusion_out_dim = self.fusion.output_dim
        else:
            self.protein_encoder = None
            self.fusion = None
            # RNA-only: project RNA output to hidden_dim
            self.rna_proj = nn.Linear(rna_out_dim, hidden_dim)
            fusion_out_dim = hidden_dim

        # Sequence encoder (Mamba + LSTM)
        if use_mamba or use_lstm:
            self.seq_encoder = MambaLSTMEncoder(
                input_dim=fusion_out_dim,
                hidden_dim=hidden_dim,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                lstm_num_layers=lstm_num_layers,
                dropout=dropout,
                use_mamba=use_mamba,
                use_lstm=use_lstm,
            )
            encoded_dim = hidden_dim
        else:
            # No sequence encoder: use mean pooling as fallback
            self.seq_encoder = None
            encoded_dim = fusion_out_dim

        # Heads
        self.classification_head = ClassificationHead(encoded_dim, num_classes, hidden_dim // 2, dropout)
        self.pseudotime_head = PseudotimeHead(encoded_dim, hidden_dim // 2, dropout)
        self.embedding_head = EmbeddingHead(encoded_dim, embedding_dim, dropout)

    def _encode_rna(self, x_rna: torch.Tensor) -> torch.Tensor:
        """Encode RNA features. Handles both 2D (B,D) and 3D (B,L,D) input."""
        if x_rna.dim() == 3:
            B, L, D = x_rna.shape
            x_flat = x_rna.reshape(B * L, D)
            z_flat = self.rna_encoder(x_flat)
            z = z_flat.reshape(B, L, -1)
        else:
            z = self.rna_encoder(x_rna)
        return z

    def _encode_protein(self, x_protein: torch.Tensor) -> torch.Tensor:
        """Encode protein features. Handles both 2D (B,D) and 3D (B,L,D) input."""
        if x_protein.dim() == 3:
            B, L, D = x_protein.shape
            x_flat = x_protein.reshape(B * L, D)
            z_flat = self.protein_encoder(x_flat)
            z = z_flat.reshape(B, L, -1)
        else:
            z = self.protein_encoder(x_protein)
        return z

    def _fuse(self, z_rna: torch.Tensor, z_protein: torch.Tensor):
        """Fuse RNA and protein representations. Handles both 2D and 3D input."""
        if z_rna.dim() == 3:
            B, L, D_rna = z_rna.shape
            D_prot = z_protein.shape[-1]
            z_rna_flat = z_rna.reshape(B * L, D_rna)
            z_prot_flat = z_protein.reshape(B * L, D_prot)
            z_fused_flat, weights_flat = self.fusion(z_rna_flat, z_prot_flat)
            z_fused = z_fused_flat.reshape(B, L, -1)
            weights = weights_flat.reshape(B, L, -1) if weights_flat is not None else None
        else:
            z_fused, weights = self.fusion(z_rna, z_protein)
        return z_fused, weights

    def forward(self, x_rna: torch.Tensor, x_protein: torch.Tensor = None):
        """
        Args:
            x_rna: RNA expression features
                   - Sequence mode: (batch_size, seq_len, rna_dim)
                   - Single mode:   (batch_size, rna_dim)
            x_protein: Protein expression features
                   - Sequence mode: (batch_size, seq_len, protein_dim)
                   - Single mode:   (batch_size, protein_dim)
                   - None: RNA-only mode

        Returns:
            dict with keys: logits, pseudotime_pred, embedding, modality_weights
        """
        is_sequence = x_rna.dim() == 3

        # ── Encode RNA ──
        z_rna = self._encode_rna(x_rna)

        # ── Encode Protein & Fuse ──
        if self.use_protein and x_protein is not None:
            z_protein = self._encode_protein(x_protein)
            z_fused, modality_weights = self._fuse(z_rna, z_protein)
        else:
            # RNA-only: project RNA to hidden_dim
            if is_sequence:
                B, L, D = z_rna.shape
                z_flat = z_rna.reshape(B * L, D)
                z_proj_flat = self.rna_proj(z_flat)
                z_fused = z_proj_flat.reshape(B, L, -1)
            else:
                z_fused = self.rna_proj(z_rna)
            modality_weights = None

        # ── Sequence Encode ──
        if self.seq_encoder is not None:
            if not is_sequence:
                # Single-cell mode: add dummy sequence dimension
                z_fused = z_fused.unsqueeze(1)  # (B, 1, hidden_dim)
            h_pooled, _ = self.seq_encoder(z_fused)
        else:
            # No sequence encoder: mean-pool over sequence dim or pass-through
            if is_sequence:
                h_pooled = z_fused.mean(dim=1)  # (B, hidden_dim)
            else:
                h_pooled = z_fused

        # ── Heads ──
        logits = self.classification_head(h_pooled)
        pseudotime_pred = self.pseudotime_head(h_pooled)
        embedding = self.embedding_head(h_pooled)

        return {
            "logits": logits,
            "pseudotime_pred": pseudotime_pred,
            "embedding": embedding,
            "modality_weights": modality_weights,
        }

    def get_mamba_backend(self):
        """Return the Mamba backend name for audit purposes."""
        if self.seq_encoder is not None and hasattr(self.seq_encoder, 'mamba'):
            if self.seq_encoder.mamba is not None:
                return self.seq_encoder.mamba.get_backend()
        return "none"

    @property
    def is_sequence_model(self):
        """Whether this model instance operates in sequence mode."""
        return self.seq_encoder is not None
