"""Multimodal fusion model for RNA + Protein data.

Supports: rna_only, protein_only, rna_protein.
Fusion modes: concat, gated.
"""
import torch
import torch.nn as nn
from .sequence_encoders import MambaLSTMEncoder
from .task_heads import build_head


class ProteinEncoder(nn.Module):
    """Simple MLP encoder for protein data."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalFusionModel(nn.Module):
    """RNA + Protein fusion model for CITE-seq data.

    Architecture:
      RNA encoder: Mamba -> LSTM (treats genes as feature sequence)
      Protein encoder: MLP
      Fusion: concat or gated
      Head: classification/regression
    """

    def __init__(
        self,
        rna_input_dim=1,
        rna_num_genes=2000,
        protein_input_dim=228,
        hidden_dim=128,
        num_mamba_blocks=2,
        lstm_layers=1,
        bidirectional=True,
        output_dim=10,
        task_type="classification",
        input_mode="rna_protein",
        fusion_mode="gated",
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        self.task_type = task_type
        self.input_mode = input_mode
        self.fusion_mode = fusion_mode

        self.has_rna = input_mode in ("rna_only", "rna_protein")
        self.has_protein = input_mode in ("protein_only", "rna_protein")

        if self.has_rna:
            self.rna_encoder = MambaLSTMEncoder(
                input_dim=rna_input_dim,
                hidden_dim=hidden_dim,
                num_mamba_blocks=num_mamba_blocks,
                lstm_layers=lstm_layers,
                bidirectional=bidirectional,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                pooling=pooling,
            )
            rna_out = self.rna_encoder.lstm_out_dim
            self.is_real_mamba = self.rna_encoder.is_real_mamba
        else:
            rna_out = 0
            self.is_real_mamba = False

        if self.has_protein:
            self.protein_encoder = ProteinEncoder(
                protein_input_dim, hidden_dim, dropout
            )
            protein_out = hidden_dim
        else:
            protein_out = 0

        if input_mode == "rna_only":
            fusion_dim = rna_out
        elif input_mode == "protein_only":
            fusion_dim = protein_out
        else:
            fusion_dim = rna_out + protein_out

        if fusion_mode == "gated" and input_mode == "rna_protein":
            self.gate = nn.Sequential(
                nn.Linear(rna_out + protein_out, hidden_dim),
                nn.Sigmoid(),
                nn.Linear(hidden_dim, rna_out),
            )

        self.head = build_head(
            task_type, fusion_dim, output_dim, dropout=dropout
        )

    def forward(self, batch):
        x = batch["x"]
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if self.input_mode == "rna_only":
            rep = self.rna_encoder(x)
            return self.head(rep)

        elif self.input_mode == "protein_only":
            protein = batch.get("protein", x.squeeze(-1))
            rep = self.protein_encoder(protein)
            return self.head(rep)

        elif self.input_mode == "rna_protein":
            rna = x
            protein = batch.get("protein")
            if protein is None:
                rep = self.rna_encoder(rna)
                return self.head(rep)

            rna_rep = self.rna_encoder(rna)
            protein_rep = self.protein_encoder(protein)

            if self.fusion_mode == "concat":
                fused = torch.cat([rna_rep, protein_rep], dim=-1)
            elif self.fusion_mode == "gated":
                combined = torch.cat([rna_rep, protein_rep], dim=-1)
                gate = self.gate(combined)
                fused = gate * rna_rep + (1 - gate) * protein_rep
            else:
                fused = torch.cat([rna_rep, protein_rep], dim=-1)

            return self.head(fused)

        else:
            raise ValueError("Unknown input_mode: " + self.input_mode)
