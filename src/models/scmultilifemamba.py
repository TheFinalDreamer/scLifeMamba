"""scMultiLifeMamba: Full three-modality trajectory-aware model.

Architecture:
    RNA -> RNAEncoder -> z_rna -|
    ATAC -> ATACEncoder -> z_atac -|-> Fusion -> z_fused
    Protein -> ProteinEncoder -> z_protein -|
                                              |
                                              v
                              MambaLSTMEncoder (trajectory sequence)
                                              |
                            +--------+--------+--------+
                            |        |        |        |
                            v        v        v        v
                          cls    ptime    branch   embedding

Supports two fusion modes:
  1. StateAwareFusion (default): static gating based on modality encodings only
  2. LagAwareDynamicFusion (use_lag_aware_fusion=True): pseudotime-aware + horizon-aware gating

The lag-aware mode is activated via forward_lag_aware() or by setting use_lag_aware_fusion=True
and passing pseudotime/horizon to forward().
"""
import torch
import torch.nn as nn
from .rna_encoder import RNAEncoder
from .atac_encoder import ATACEncoder
from .protein_encoder import ProteinEncoder
from .modality_fusion import StateAwareFusion, ConcatFusion
from .mamba_lstm import MambaLSTMEncoder
from .task_heads import ClassificationHead, RegressionHead, BranchHead, EmbeddingHead


class scMultiLifeMamba(nn.Module):
    def __init__(
        self,
        rna_dim: int,
        num_classes: int,
        atac_dim: int = 0,
        protein_dim: int = 0,
        num_branches: int = 0,
        rna_hidden_dims: list = None,
        atac_hidden_dims: list = None,
        protein_hidden_dims: list = None,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.2,
        use_atac: bool = False,
        use_protein: bool = False,
        use_mamba: bool = True,
        use_lstm: bool = True,
        use_dynamic_fusion: bool = True,
        use_transformer: bool = False,
        use_lag_aware_fusion: bool = False,
        n_pseudotime_bins: int = 20,
        max_horizon: int = 16,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        lstm_num_layers: int = 2,
    ):
        super().__init__()
        self.use_atac = use_atac
        self.use_protein = use_protein
        self.use_mamba = use_mamba
        self.use_lstm = use_lstm
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_lag_aware_fusion = use_lag_aware_fusion
        self.use_transformer = use_transformer
        self.hidden_dim = hidden_dim

        if rna_hidden_dims is None:
            rna_hidden_dims = [512, 256]
        if atac_hidden_dims is None:
            atac_hidden_dims = [256, 128]
        if protein_hidden_dims is None:
            protein_hidden_dims = [64]

        # Encoders
        self.rna_encoder = RNAEncoder(rna_dim, rna_hidden_dims, dropout, use_transformer)
        rna_out = self.rna_encoder.output_dim

        self.atac_encoder = None
        atac_out = 0
        if use_atac and atac_dim > 0:
            self.atac_encoder = ATACEncoder(atac_dim, atac_hidden_dims, dropout, use_transformer)
            atac_out = self.atac_encoder.output_dim

        self.protein_encoder = None
        prot_out = 0
        if use_protein and protein_dim > 0:
            self.protein_encoder = ProteinEncoder(protein_dim, protein_hidden_dims, dropout)
            prot_out = self.protein_encoder.output_dim

        # Fusion
        n_modalities = 1 + int(use_atac) + int(use_protein)
        self._n_modalities = n_modalities
        self.lag_aware_fusion = None

        if n_modalities > 1:
            if use_lag_aware_fusion:
                from .lag_aware_dynamic_fusion import LagAwareDynamicFusion
                self.lag_aware_fusion = LagAwareDynamicFusion(
                    d_model=hidden_dim,
                    num_modalities=n_modalities,
                    n_pseudotime_bins=n_pseudotime_bins,
                    max_horizon=max_horizon,
                    use_atac=use_atac,
                    use_protein=use_protein,
                    dropout=dropout,
                )
                # Keep static fusion as fallback
                self.fusion = ConcatFusion(
                    rna_out, atac_out if use_atac else None,
                    prot_out if use_protein else None,
                    hidden_dim, use_atac, use_protein,
                )
            elif use_dynamic_fusion:
                self.fusion = StateAwareFusion(
                    rna_out, atac_out if use_atac else None,
                    prot_out if use_protein else None,
                    hidden_dim, use_atac, use_protein,
                )
            else:
                self.fusion = ConcatFusion(
                    rna_out, atac_out if use_atac else None,
                    prot_out if use_protein else None,
                    hidden_dim, use_atac, use_protein,
                )
            fusion_out = hidden_dim
        else:
            self.fusion = None
            self.rna_proj = nn.Linear(rna_out, hidden_dim)
            fusion_out = hidden_dim

        # Sequence encoder (Mamba-LSTM over trajectory window)
        if use_mamba or use_lstm:
            self.seq_encoder = MambaLSTMEncoder(
                input_dim=fusion_out, hidden_dim=hidden_dim,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, lstm_num_layers=lstm_num_layers,
                dropout=dropout, use_mamba=use_mamba, use_lstm=use_lstm,
            )
            encoded_dim = hidden_dim
        else:
            self.seq_encoder = None
            encoded_dim = fusion_out

        # Heads
        self.classification_head = ClassificationHead(encoded_dim, num_classes, hidden_dim // 2, dropout)
        self.pseudotime_head = RegressionHead(encoded_dim, hidden_dim // 2, dropout)
        self.embedding_head = EmbeddingHead(encoded_dim, embedding_dim, hidden_dim // 2, dropout)
        self.branch_head = BranchHead(encoded_dim, num_branches, hidden_dim // 2, dropout) if num_branches > 0 else None

    def forward(self, x_rna, x_atac=None, x_protein=None, modality_mask=None,
                return_sequence=False, return_modality_weights=False,
                pseudotime=None, horizon=None, task_id=0):
        """Standard forward pass using StateAwareFusion.

        Args:
            x_rna: (B, seq_len, rna_dim) or (B, rna_dim)
            x_atac: (B, seq_len, atac_dim) or (B, atac_dim) — optional
            x_protein: (B, seq_len, protein_dim) or (B, protein_dim) — optional
            modality_mask: (B, n_modalities) — optional
            pseudotime: (B,) or (B, seq_len) — only used if lag_aware_fusion active
            horizon: int or (B,) — only used if lag_aware_fusion active
            task_id: int — only used if lag_aware_fusion active

        Returns: dict with logits, pseudotime_pred, embedding, [branch_logits, modality_weights, h_sequence]
        """
        orig_shape = x_rna.shape
        has_seq = len(orig_shape) == 3

        if has_seq:
            B, S, Dr = x_rna.shape
            x_rna_flat = x_rna.reshape(B * S, Dr)
            Da = x_atac.shape[-1] if x_atac is not None else 0
            Dp = x_protein.shape[-1] if x_protein is not None else 0
            x_atac_flat = x_atac.reshape(B * S, Da) if x_atac is not None else None
            x_protein_flat = x_protein.reshape(B * S, Dp) if x_protein is not None else None
        else:
            x_rna_flat = x_rna
            x_atac_flat = x_atac
            x_protein_flat = x_protein

        z_rna = self.rna_encoder(x_rna_flat)
        z_atac = self.atac_encoder(x_atac_flat) if self.atac_encoder is not None else None
        z_protein = self.protein_encoder(x_protein_flat) if self.protein_encoder is not None else None

        if self.lag_aware_fusion is not None:
            # Lag-aware dynamic fusion path
            z_fused_flat, mod_weights = self.lag_aware_fusion(
                z_rna=z_rna,
                z_protein=z_protein,
                z_atac=z_atac,
                pseudotime=pseudotime,
                horizon=horizon if horizon is not None else 0,
                task_id=task_id,
                modality_mask=modality_mask,
                return_weights=return_modality_weights,
            )
        elif self.fusion is not None:
            z_fused_flat, mod_weights = self.fusion(z_rna, z_atac, z_protein, modality_mask)
        else:
            z_fused_flat = self.rna_proj(z_rna)
            mod_weights = torch.ones(z_rna.size(0), 1, device=z_rna.device)

        if has_seq:
            z_fused = z_fused_flat.reshape(B, S, -1) if z_fused_flat.dim() == 2 else z_fused_flat
        else:
            z_fused = z_fused_flat

        if self.seq_encoder is not None:
            if not has_seq:
                z_fused = z_fused.unsqueeze(1)
            h_pooled, h_seq = self.seq_encoder(z_fused)
        else:
            h_pooled = z_fused[:, z_fused.size(1) // 2, :] if has_seq else z_fused
            h_seq = z_fused

        out = {}
        out.update(self.classification_head(h_pooled))
        out.update(self.pseudotime_head(h_pooled))
        out.update(self.embedding_head(h_pooled))
        if self.branch_head is not None:
            out.update(self.branch_head(h_pooled))
        if return_modality_weights:
            out["modality_weights"] = mod_weights
        if return_sequence:
            out["h_sequence"] = h_seq
        return out

    def forward_lag_aware(self, x_rna, pseudotime, horizon, x_atac=None, x_protein=None,
                          task_id=0, modality_mask=None, return_sequence=False):
        """Dedicated lag-aware forward for cross-modal temporal prediction.

        Uses LagAwareDynamicFusion with pseudotime and horizon embeddings.
        Falls back silently to static fusion if lag_aware_fusion is None.

        Args:
            x_rna: (B, seq_len, rna_dim) or (B, rna_dim)
            pseudotime: (B,) or (B, seq_len) — pseudotime value for gating
            horizon: int or (B,) — prediction horizon
            x_atac: (B, seq_len, atac_dim) — optional
            x_protein: (B, seq_len, protein_dim) — optional
            task_id: 0=rna_to_protein, 1=protein_to_rna, 2=joint_to_state, 3=joint_to_pseudotime
            modality_mask: (B, n_modalities) — optional

        Returns: dict with same keys as forward() plus modality_weights
        """
        return self.forward(
            x_rna=x_rna,
            x_atac=x_atac,
            x_protein=x_protein,
            modality_mask=modality_mask,
            return_sequence=return_sequence,
            return_modality_weights=True,
            pseudotime=pseudotime,
            horizon=horizon,
            task_id=task_id,
        )

    def get_model_summary(self):
        """Return model summary dict for logging."""
        summary = {
            "model": "scMultiLifeMamba",
            "parameters": self.count_parameters(),
            "trainable_parameters": self.count_trainable_parameters(),
            "n_modalities": self.n_modalities,
            "use_atac": self.use_atac,
            "use_protein": self.use_protein,
            "use_mamba": self.use_mamba,
            "use_lstm": self.use_lstm,
            "use_lag_aware_fusion": self.use_lag_aware_fusion,
            "is_real_mamba": self.is_real_mamba,
            "hidden_dim": self.hidden_dim,
            "fusion_type": "lag_aware_dynamic" if self.lag_aware_fusion is not None
                           else ("state_aware" if self.use_dynamic_fusion else "concat"),
        }
        if self.lag_aware_fusion is not None:
            summary["lag_aware_config"] = {
                "n_pseudotime_bins": self.lag_aware_fusion.n_pseudotime_bins,
                "max_horizon": self.lag_aware_fusion.max_horizon,
            }
        return summary

    @property
    def is_real_mamba(self):
        if self.seq_encoder is not None:
            return self.seq_encoder.is_real_mamba
        return False

    @property
    def n_modalities(self):
        return getattr(self, '_n_modalities', 1 + int(self.use_atac) + int(self.use_protein))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
