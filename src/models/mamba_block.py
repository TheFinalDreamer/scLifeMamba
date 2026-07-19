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
    """Unified Mamba-style block with validated backend selection.

    Priority:
      1. RealMambaBlock (mamba_ssm) — requires mamba_ssm package (Linux)
      2. TorchSelectiveSSMBlock — PyTorch-native selective SSM (cross-platform, validated)
      3. FallbackMambaBlock — Conv1d+GRU (diagnostic/engineering only, NOT for formal results)
    """

    def __init__(self, dim=128, d_state=16, d_conv=4, expand=2, dropout=0.1, require_native=False):
        super().__init__()
        self.dim = dim
        self.backend_name = None

        if is_mamba_available():
            self._block = RealMambaBlock(dim, d_state, d_conv, expand)
            self.backend_name = "native_mamba_ssm"
        else:
            # Use validated PyTorch-native selective SSM (NOT fallback)
            try:
                from .torch_selective_ssm import TorchSelectiveSSMBlock
                self._block = TorchSelectiveSSMBlock(dim, d_state, d_conv, expand, dropout)
                self.backend_name = "torch_selective_ssm"
            except ImportError:
                if require_native:
                    raise RuntimeError(
                        "No selective SSM backend available: "
                        "mamba_ssm not installed, torch_selective_ssm not importable. "
                        "Cannot use fallback when require_native=True."
                    )
                self._block = FallbackMambaBlock(dim, d_state, d_conv, expand, dropout)
                self.backend_name = "fallback_conv1d_gru"

    def forward(self, x):
        return self._block(x)

    def get_backend(self):
        return self.backend_name
