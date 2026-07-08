"""RNAEncoder: RNA expression encoder with MLP/Transformer and optional scGPT input."""
import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: list, dropout: float) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = h_dim
    return nn.Sequential(*layers)


class RNAEncoder(nn.Module):
    """Encode RNA expression (raw/HVG/PCA/scGPT) into latent representation."""

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2,
                 use_transformer: bool = False, nhead: int = 4, n_layers: int = 2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.mlp = _build_mlp(input_dim, hidden_dims, dropout)
        mlp_out = hidden_dims[-1] if hidden_dims else input_dim

        self.use_transformer = use_transformer
        if use_transformer:
            self.proj_in = nn.Linear(mlp_out, mlp_out)
            encoder_layer = nn.TransformerEncoderLayer(d_model=mlp_out, nhead=nhead,
                                                       dim_feedforward=mlp_out * 4,
                                                       dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_dim = mlp_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        if self.use_transformer:
            h = self.proj_in(h)
            h = h.unsqueeze(1)
            h = self.transformer(h)
            h = h.squeeze(1)
        return h
