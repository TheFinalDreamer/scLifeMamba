"""
Baseline model implementations for comparison with scLifeMamba.

All baselines share the same forward output format:
{
    "logits": logits,
    "pseudotime_pred": pseudotime_pred,
    "embedding": embedding,
    "modality_weights": None
}

Supports RNA+Protein and RNA-only modes via use_protein flag.
"""

import torch
import torch.nn as nn

from .encoders import RNAEncoder, ProteinEncoder
from .fusion import StateAwareFusion, SimpleFusion
from .heads import ClassificationHead, PseudotimeHead, EmbeddingHead


class _BaseBaseline(nn.Module):
    """Shared backbone for all baselines: encode -> fusion -> process -> heads."""

    def __init__(
        self,
        rna_dim: int,
        protein_dim: int,
        num_classes: int,
        rna_hidden_dims: list,
        protein_hidden_dims: list,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.2,
        use_dynamic_fusion: bool = False,
        use_protein: bool = True,
    ):
        super().__init__()
        self.use_protein = use_protein

        self.rna_encoder = RNAEncoder(rna_dim, rna_hidden_dims, dropout)
        rna_out = self.rna_encoder.output_dim

        if use_protein:
            self.protein_encoder = ProteinEncoder(protein_dim, protein_hidden_dims, dropout)
            protein_out = self.protein_encoder.output_dim

            if use_dynamic_fusion:
                self.fusion = StateAwareFusion(rna_out, protein_out, hidden_dim)
            else:
                self.fusion = SimpleFusion(rna_out, protein_out, hidden_dim)

            self.fusion_out_dim = self.fusion.output_dim
        else:
            self.protein_encoder = None
            self.fusion = None
            self.rna_proj = nn.Linear(rna_out, hidden_dim)
            self.fusion_out_dim = hidden_dim

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_classes = num_classes

    def build_heads(self, input_dim: int):
        """Build prediction heads. Called after encoder is defined."""
        self.classification_head = ClassificationHead(input_dim, self.num_classes, self.hidden_dim // 2, self.dropout)
        self.pseudotime_head = PseudotimeHead(input_dim, self.hidden_dim // 2, self.dropout)
        self.embedding_head = EmbeddingHead(input_dim, self.embedding_dim, self.dropout)

    def _forward_heads(self, h: torch.Tensor):
        return {
            "logits": self.classification_head(h),
            "pseudotime_pred": self.pseudotime_head(h),
            "embedding": self.embedding_head(h),
            "modality_weights": None,
        }

    def forward_encoders(self, x_rna, x_protein=None):
        z_rna = self.rna_encoder(x_rna)
        if self.use_protein and x_protein is not None:
            z_protein = self.protein_encoder(x_protein)
            z_fused, _ = self.fusion(z_rna, z_protein)
        else:
            z_fused = self.rna_proj(z_rna)
        return z_fused


class MLPBaseline(_BaseBaseline):
    """Simple MLP baseline: encode -> concat-fuse -> MLP -> heads."""

    def __init__(self, rna_dim, protein_dim, num_classes, **kwargs):
        super().__init__(rna_dim, protein_dim, num_classes, **kwargs)
        self.process = nn.Sequential(
            nn.Linear(self.fusion_out_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.build_heads(self.hidden_dim)

    def forward(self, x_rna, x_protein=None):
        z_fused = self.forward_encoders(x_rna, x_protein)
        h = self.process(z_fused)
        return self._forward_heads(h)


class LSTMBaseline(_BaseBaseline):
    """LSTM baseline: encode -> concat-fuse -> LSTM -> heads."""

    def __init__(self, rna_dim, protein_dim, num_classes, lstm_num_layers=2, **kwargs):
        super().__init__(rna_dim, protein_dim, num_classes, **kwargs)
        self.lstm = nn.LSTM(
            input_size=self.fusion_out_dim,
            hidden_size=self.hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=self.dropout if lstm_num_layers > 1 else 0.0,
        )
        self.build_heads(self.hidden_dim)

    def forward(self, x_rna, x_protein=None):
        z_fused = self.forward_encoders(x_rna, x_protein)
        z_seq = z_fused.unsqueeze(1)
        h_lstm, _ = self.lstm(z_seq)
        h = h_lstm.squeeze(1)
        return self._forward_heads(h)


class TransformerBaseline(_BaseBaseline):
    """Transformer baseline: encode -> concat-fuse -> TransformerEncoder -> heads."""

    def __init__(self, rna_dim, protein_dim, num_classes,
                 transformer_num_heads=4, transformer_num_layers=2, **kwargs):
        super().__init__(rna_dim, protein_dim, num_classes, **kwargs)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_out_dim,
            nhead=transformer_num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        self.build_heads(self.fusion_out_dim)

    def forward(self, x_rna, x_protein=None):
        z_fused = self.forward_encoders(x_rna, x_protein)
        z_seq = z_fused.unsqueeze(1)
        h_trans = self.transformer(z_seq)
        h = h_trans.squeeze(1)
        return self._forward_heads(h)


class MambaOnlyBaseline(_BaseBaseline):
    """Mamba-only baseline: encode -> concat-fuse -> MambaBlock -> heads."""

    def __init__(self, rna_dim, protein_dim, num_classes,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, **kwargs):
        super().__init__(rna_dim, protein_dim, num_classes, **kwargs)
        from .mamba_block import MambaBlock
        self.mamba = MambaBlock(
            dim=self.fusion_out_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.build_heads(self.fusion_out_dim)

    def forward(self, x_rna, x_protein=None):
        z_fused = self.forward_encoders(x_rna, x_protein)
        z_seq = z_fused.unsqueeze(1)
        h_mamba = self.mamba(z_seq)
        h = h_mamba.squeeze(1)
        return self._forward_heads(h)


class LSTMOnlyBaseline(LSTMBaseline):
    """LSTM-only baseline: alias for LSTMBaseline."""
    pass
