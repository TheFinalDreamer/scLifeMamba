"""Model factory: build scMultiLifeMamba variants and baselines from config."""
import torch.nn as nn
from .scmultilifemamba import scMultiLifeMamba
from .rna_encoder import RNAEncoder
from .task_heads import ClassificationHead, RegressionHead, BranchHead


def create_scmultilifemamba(config: dict) -> scMultiLifeMamba:
    model_cfg = config.get("model", config)
    return scMultiLifeMamba(
        rna_dim=model_cfg.get("rna_dim", 2000),
        num_classes=model_cfg.get("num_classes", 10),
        atac_dim=model_cfg.get("atac_dim", 0),
        protein_dim=model_cfg.get("protein_dim", 0),
        num_branches=model_cfg.get("num_branches", 0),
        rna_hidden_dims=model_cfg.get("rna_hidden_dims"),
        atac_hidden_dims=model_cfg.get("atac_hidden_dims"),
        protein_hidden_dims=model_cfg.get("protein_hidden_dims"),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        embedding_dim=model_cfg.get("embedding_dim", 64),
        dropout=model_cfg.get("dropout", 0.2),
        use_atac=model_cfg.get("use_atac", False),
        use_protein=model_cfg.get("use_protein", False),
        use_mamba=model_cfg.get("use_mamba", True),
        use_lstm=model_cfg.get("use_lstm", True),
        use_dynamic_fusion=model_cfg.get("use_dynamic_fusion", True),
        use_transformer=model_cfg.get("use_transformer", False),
        mamba_d_state=model_cfg.get("mamba_d_state", 16),
        mamba_d_conv=model_cfg.get("mamba_d_conv", 4),
        mamba_expand=model_cfg.get("mamba_expand", 2),
        lstm_num_layers=model_cfg.get("lstm_num_layers", 2),
    )


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.head = ClassificationHead(hidden_dims[-1], num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, x.size(1) // 2, :]
        h = self.encoder(x)
        return self.head(h)


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.head = ClassificationHead(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        h = out[:, out.size(1) // 2, :]
        return self.head(h)


class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, nhead=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = ClassificationHead(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.transformer(x)
        h = out[:, out.size(1) // 2, :]
        return self.head(h)


def build_model(model_name, input_dim=None, num_classes=10, hidden_dim=128, device="cpu", **kwargs):
    """Unified factory for model building."""
    config = kwargs.get("config", {})
    model_cfg = config.get("model", config)
    rna_dim = model_cfg.get("rna_dim", input_dim or 2000)
    n_cls = model_cfg.get("num_classes", num_classes)

    if model_name == "scmultilifemamba" or model_name == "scLifeMamba":
        return create_scmultilifemamba(config)
    elif model_name == "mlp":
        d = model_cfg.get("input_dim", input_dim or rna_dim)
        return MLPBaseline(d, n_cls)
    elif model_name == "lstm":
        d = model_cfg.get("input_dim", input_dim or rna_dim)
        return LSTMBaseline(d, n_cls, hidden_dim=hidden_dim)
    elif model_name == "transformer":
        d = model_cfg.get("input_dim", input_dim or rna_dim)
        return TransformerBaseline(d, n_cls, hidden_dim=hidden_dim)
    else:
        return create_scmultilifemamba(config)
