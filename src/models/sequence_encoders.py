"""Sequence encoders: MambaOnly, LSTMOnly, MambaLSTM.

All encoders accept (B, L, 1) input and return (B, D) pooled representation.
"""
import torch
import torch.nn as nn
from .mamba_block import create_mamba_block
from .task_heads import build_head


class MambaOnlyModel(nn.Module):
    """Stacked Mamba blocks + pooling + head."""

    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_blocks=2,
        output_dim=10,
        task_type="classification",
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        self.task_type = task_type
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            create_mamba_block(hidden_dim, d_state, d_conv, expand, dropout)
            for _ in range(num_blocks)
        ])
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = build_head(task_type, hidden_dim, output_dim, dropout=dropout)
        self.is_real_mamba = getattr(self.blocks[0], "is_real_mamba", False)

    def forward(self, batch):
        x = batch["x"]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        if self.pooling == "mean":
            rep = x.mean(dim=1)
        elif self.pooling == "max":
            rep = x.max(dim=1).values
        else:
            rep = x[:, -1, :]
        rep = self.dropout(rep)
        return self.head(rep)


class LSTMOnlyModel(nn.Module):
    """Bidirectional LSTM encoder + head."""

    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        output_dim=10,
        task_type="classification",
        dropout=0.1,
        bidirectional=True,
        pooling="mean",
    ):
        super().__init__()
        self.task_type = task_type
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_dim * 2 if bidirectional else hidden_dim
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = build_head(task_type, lstm_out, output_dim, dropout=dropout)

    def forward(self, batch):
        x = batch["x"]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        if self.pooling == "mean":
            rep = out.mean(dim=1)
        elif self.pooling == "max":
            rep = out.max(dim=1).values
        else:
            rep = out[:, -1, :]
        rep = self.dropout(rep)
        return self.head(rep)


class MambaLSTMEncoder(nn.Module):
    """Mamba -> LSTM hybrid encoder. Returns pooled representation."""

    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_mamba_blocks=2,
        lstm_layers=1,
        bidirectional=True,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba_blocks = nn.ModuleList([
            create_mamba_block(hidden_dim, d_state, d_conv, expand, dropout)
            for _ in range(num_mamba_blocks)
        ])
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.is_real_mamba = getattr(self.mamba_blocks[0], "is_real_mamba", False)

    def forward(self, x):
        # x: (B, L, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)
        for blk in self.mamba_blocks:
            x = blk(x)
        out, _ = self.lstm(x)
        if self.pooling == "mean":
            rep = out.mean(dim=1)
        elif self.pooling == "max":
            rep = out.max(dim=1).values
        else:
            rep = out[:, -1, :]
        rep = self.dropout(rep)
        return rep


class MambaLSTMModel(nn.Module):
    """Mamba -> LSTM hybrid model with task head."""

    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_mamba_blocks=2,
        lstm_layers=1,
        output_dim=10,
        task_type="classification",
        bidirectional=True,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__()
        self.task_type = task_type
        self.encoder = MambaLSTMEncoder(
            input_dim=input_dim,
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
        self.head = build_head(
            task_type, self.encoder.lstm_out_dim, output_dim, dropout=dropout
        )
        self.is_real_mamba = self.encoder.is_real_mamba

    def forward(self, batch):
        x = batch["x"]
        rep = self.encoder(x)
        return self.head(rep)
