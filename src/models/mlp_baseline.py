"""MLP baseline compatible with model_factory interface.

Accepts batch dict with "x" key, returns task-appropriate output dict.
"""

import torch.nn as nn
from .task_heads import build_head


class MLPBaseline(nn.Module):
    """Simple MLP baseline for classification / regression / multitask."""

    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 64),
        output_dim=10,
        task_type="classification",
        dropout=0.2,
    ):
        super().__init__()
        self.task_type = task_type
        self.flatten = nn.Flatten()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = build_head(task_type, in_dim, output_dim, hidden_dim=in_dim // 2, dropout=dropout)

    def forward(self, batch):
        x = batch["x"]
        # Support both (B, D) and (B, L, D) — flatten if 3D
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        h = self.backbone(x)
        return self.head(h)
