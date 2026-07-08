"""MambaBlock: Real mamba-ssm or fallback, with consistent interface."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def is_mamba_available():
    try:
        import mamba_ssm  # noqa: F401
        return True
    except ImportError:
        return False


def get_mamba_version():
    try:
        import mamba_ssm
        return getattr(mamba_ssm, '__version__', 'unknown')
    except ImportError:
        return None


class RealMambaBlock(nn.Module):
    """Wrapper around mamba_ssm.Mamba."""

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.d_model = d_model

    def forward(self, x):
        return self.mamba(x)


class FallbackMambaBlock(nn.Module):
    """Fallback: depthwise Conv1d + gated projection + GRU approximating Mamba."""

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        # Use symmetric padding to preserve sequence length for even kernels
        self.d_conv = d_conv
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                              padding=0, groups=d_inner)
        self.gate_proj = nn.Linear(d_model, d_inner)
        self.gru = nn.GRU(d_inner, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        projected = self.in_proj(x)
        u, z = projected.chunk(2, dim=-1)
        u_t = u.transpose(1, 2)  # [B, d_inner, L]
        # Symmetric padding to preserve sequence length
        pad = self.d_conv - 1
        u_t = F.pad(u_t, (pad // 2, pad - pad // 2))
        u_t = self.conv(u_t)
        u = F.silu(u_t.transpose(1, 2))
        gate = torch.sigmoid(self.gate_proj(x))
        u = u * gate
        out, _ = self.gru(u)
        out = self.norm(out + self.out_proj(residual))
        out = self.dropout(out)
        return out


class MambaBlock(nn.Module):
    """Unified Mamba block: real if available, fallback otherwise."""

    def __init__(self, dim=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.is_real_mamba = is_mamba_available()
        if self.is_real_mamba:
            self._block = RealMambaBlock(dim, d_state, d_conv, expand)
        else:
            self._block = FallbackMambaBlock(dim, d_state, d_conv, expand, dropout)
        self.dim = dim

    def forward(self, x):
        return self._block(x)
