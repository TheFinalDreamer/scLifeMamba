"""ATACEncoder: ATAC-seq encoder supporting LSI, gene activity, or peak matrix input."""
import torch
import torch.nn as nn


class ATACEncoder(nn.Module):
    """Encode ATAC features (LSI components, gene activity scores, or peak matrix).

    Architecture: MLP with optional transformer layers.
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2,
                 use_transformer: bool = False, nhead: int = 4, n_layers: int = 2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
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
