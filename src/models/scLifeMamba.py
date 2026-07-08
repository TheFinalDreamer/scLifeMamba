"""
scLifeMamba: Mamba-LSTM Multi-modal Fusion Model for Single-Cell Analysis.

Full architecture (RNA+Protein):
    x_rna -> RNAEncoder -> z_rna -|
                                   |-> StateAwareFusion -> z_fused
    x_protein -> ProteinEncoder -> z_protein -|
                                                    |
                                                    v
                                    MambaLSTMEncoder -> h_encoded
                                                    |
                                     +--------------+--------------+
                                     |              |              |
                                     v              v              v
                              ClassificationHead  PseudotimeHead  EmbeddingHead

RNA-only mode (use_protein=False):
    x_rna -> RNAEncoder -> z_rna -> MambaLSTMEncoder -> h_encoded -> heads
"""

import torch
import torch.nn as nn

from .encoders import RNAEncoder, ProteinEncoder
from .fusion import StateAwareFusion, SimpleFusion
from .mamba_lstm import MambaLSTMEncoder
from .heads import ClassificationHead, PseudotimeHead, EmbeddingHead


class scLifeMamba(nn.Module):
    """Mamba-LSTM multi-modal fusion model for single-cell continuous state modeling.

    Supports RNA+Protein and RNA-only modes.

    Inputs:
        - x_rna: RNA expression features (B, rna_dim)
        - x_protein: Protein expression features (B, protein_dim) — ignored if use_protein=False

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
            )
            encoded_dim = hidden_dim
        else:
            self.seq_encoder = nn.Identity()
            encoded_dim = fusion_out_dim

        # Heads
        self.classification_head = ClassificationHead(encoded_dim, num_classes, hidden_dim // 2, dropout)
        self.pseudotime_head = PseudotimeHead(encoded_dim, hidden_dim // 2, dropout)
        self.embedding_head = EmbeddingHead(encoded_dim, embedding_dim, dropout)

    def forward(self, x_rna: torch.Tensor, x_protein: torch.Tensor = None):
        """
        Args:
            x_rna: (batch_size, rna_dim)
            x_protein: (batch_size, protein_dim) — optional, ignored if use_protein=False

        Returns:
            dict with keys: logits, pseudotime_pred, embedding, modality_weights
        """
        # Encode RNA
        z_rna = self.rna_encoder(x_rna)

        if self.use_protein and x_protein is not None:
            # Encode protein and fuse
            z_protein = self.protein_encoder(x_protein)
            z_fused, modality_weights = self.fusion(z_rna, z_protein)
        else:
            # RNA-only: project RNA to hidden_dim
            z_fused = self.rna_proj(z_rna)
            modality_weights = None

        # Sequence encode (Mamba + LSTM or pass-through)
        h = self.seq_encoder(z_fused)

        # Heads
        logits = self.classification_head(h)
        pseudotime_pred = self.pseudotime_head(h)
        embedding = self.embedding_head(h)

        return {
            "logits": logits,
            "pseudotime_pred": pseudotime_pred,
            "embedding": embedding,
            "modality_weights": modality_weights,
        }
