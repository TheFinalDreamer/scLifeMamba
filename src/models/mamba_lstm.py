"""MambaLSTMEncoder: Mamba + LSTM for trajectory-aware sequence encoding."""
import torch
import torch.nn as nn
from .mamba_block import MambaBlock


class MambaLSTMEncoder(nn.Module):
    """Mamba -> LSTM with residual/gated fusion for trajectory sequences.

    Input: (B, seq_len, hidden_dim) trajectory-aware cell state sequence.
    h_mamba = Mamba(x)
    h_lstm, _ = LSTM(x)
    h = h_mamba + gamma * h_lstm
    h_pooled = pool(h) — center-pooling or attention-pooling.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2,
                 lstm_num_layers: int = 2, dropout: float = 0.2,
                 use_mamba: bool = True, use_lstm: bool = True,
                 pool_type: str = "center"):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_lstm = use_lstm
        self.pool_type = pool_type

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_mamba:
            self.mamba = MambaBlock(dim=hidden_dim, d_state=mamba_d_state,
                                    d_conv=mamba_d_conv, expand=mamba_expand, dropout=dropout)
        else:
            self.mamba = None

        if use_lstm:
            self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                                num_layers=lstm_num_layers, batch_first=True,
                                dropout=dropout if lstm_num_layers > 1 else 0.0)
        else:
            self.lstm = None

        self.gamma = nn.Parameter(torch.tensor(0.5)) if (use_mamba and use_lstm) else None
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if pool_type == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1),
            )

    def forward(self, x: torch.Tensor):
        """Args:
            x: (B, seq_len, hidden_dim) trajectory sequence

        Returns:
            h_pooled: (B, hidden_dim) pooled representation
        """
        h = self.input_proj(x)

        h_mamba = self.mamba(h) if self.use_mamba else h
        h_lstm, _ = self.lstm(h) if self.use_lstm else (h, None)

        if self.use_mamba and self.use_lstm:
            h_out = h_mamba + self.gamma * h_lstm
        elif self.use_mamba:
            h_out = h_mamba
        elif self.use_lstm:
            h_out = h_lstm
        else:
            h_out = h

        if self.pool_type == "center":
            center_idx = h_out.size(1) // 2
            h_pooled = h_out[:, center_idx, :]
        elif self.pool_type == "attention":
            weights = self.attn_pool(h_out)
            h_pooled = (h_out * weights).sum(dim=1)
        elif self.pool_type == "mean":
            h_pooled = h_out.mean(dim=1)
        elif self.pool_type == "last":
            h_pooled = h_out[:, -1, :]
        else:
            h_pooled = h_out[:, h_out.size(1) // 2, :]

        h_pooled = self.norm(h_pooled)
        h_pooled = self.dropout(h_pooled)
        return h_pooled, h_out

    @property
    def is_real_mamba(self):
        if self.mamba is not None:
            return getattr(self.mamba, 'is_real_mamba', False)
        return False
