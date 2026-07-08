"""Task heads: classification, pseudotime, branch prediction, embedding, and factory."""
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return {"logits": logits, "pred": logits.argmax(dim=-1)}


class RegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        pred = self.net(x).squeeze(-1)
        return {"pred": pred, "pseudotime": pred}


class BranchHead(nn.Module):
    """Predict trajectory branch assignment."""
    def __init__(self, input_dim, num_branches, hidden_dim=None, dropout=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_branches),
        )

    def forward(self, x):
        logits = self.net(x)
        return {"logits": logits, "pred": logits.argmax(dim=-1)}


class EmbeddingHead(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return {"embedding": self.net(x)}


class MultiTaskHead(nn.Module):
    """Combined classification + regression + branch head."""
    def __init__(self, input_dim, n_classes, num_branches=0, hidden_dim=None, dropout=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_dim, n_classes)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.has_branch = num_branches > 0
        if self.has_branch:
            self.branch_head = nn.Linear(hidden_dim, num_branches)

    def forward(self, x):
        h = self.shared(x)
        out = {
            "logits": self.cls_head(h),
            "pred": self.cls_head(h).argmax(dim=-1),
            "pseudotime": self.reg_head(h).squeeze(-1),
        }
        if self.has_branch:
            branch_logits = self.branch_head(h)
            out["branch_logits"] = branch_logits
            out["branch_pred"] = branch_logits.argmax(dim=-1)
        return out


def build_head(head_type, input_dim, n_classes=None, hidden_dim=None, dropout=0.2):
    if head_type == "classification":
        return ClassificationHead(input_dim, n_classes or 2, hidden_dim, dropout)
    elif head_type == "regression":
        return RegressionHead(input_dim, hidden_dim, dropout)
    elif head_type == "multitask":
        return MultiTaskHead(input_dim, n_classes or 2, 0, hidden_dim, dropout)
    else:
        raise ValueError("Unknown head_type: " + str(head_type))
