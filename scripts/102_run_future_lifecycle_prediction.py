#!/usr/bin/env python3
"""
102_run_future_lifecycle_prediction.py
Future lifecycle stage prediction experiment.
Supports 6 models, multi-seed, multi-horizon, multi-context.
Outputs to outputs/lifecycle_prediction/lifecycle_prediction/

Usage:
  python code/scripts/102_run_future_lifecycle_prediction.py --model mamba_lstm --horizon 4 --seed 42
  python code/scripts/102_run_future_lifecycle_prediction.py --model all --horizon 4 --seed 42
"""
import os, sys, json
from pathlib import Path, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "lifecycle_prediction")
os.makedirs(OUT_DIR, exist_ok=True)

# Model registry
MODELS = ["mlp", "lstm", "transformer", "mamba", "mamba_lstm", "lag_aware_fusion"]

def get_model_dir(model_name, horizon, seed, context_window):
    d = os.path.join(OUT_DIR, model_name, f"h{horizon}_s{seed}_ctx{context_window}")
    os.makedirs(d, exist_ok=True)
    return d


def load_data(context_window=8, horizon=4):
    """Load trajectory sequences and lifecycle labels."""
    import scanpy as sc

    # Try to load trajectories
    traj_paths = [
        os.path.join(PROJECT_ROOT, "outputs/highdim_real_mamba/data/trajectory_data.h5ad"),
        os.path.join(PROJECT_ROOT, "data/processed/trajectory_sequences.h5ad"),
    ]
    adata = None
    for p in traj_paths:
        if os.path.exists(p):
            adata = sc.read_h5ad(p)
            break

    if adata is None:
        raise FileNotFoundError("No trajectory data found")

    # Load labels
    label_path = os.path.join(PROJECT_ROOT, "outputs/lifecycle_prediction/lifecycle_labels/labels_4bin.csv")
    if os.path.exists(label_path):
        labels_df = pd.read_csv(label_path)
        lifecycle_labels = labels_df["lifecycle_stage"].values
    else:
        # Compute from pseudotime
        pt = adata.obs["pseudotime"].values if "pseudotime" in adata.obs else adata.obs["dpt_pseudotime"].values
        pt_norm = (pt - pt.min()) / (pt.max() - pt.min()) if pt.max() > pt.min() else pt
        lifecycle_labels = np.floor(pt_norm * 4).astype(int)
        lifecycle_labels = np.clip(lifecycle_labels, 0, 3)

    # Build sequences
    n_cells = adata.n_obs
    rna_key = [k for k in adata.obsm.keys() if "rna" in k.lower() or "X_pca" in k.lower()]
    prot_key = [k for k in adata.obsm.keys() if "prot" in k.lower() or "protein" in k.lower() or "adt" in k.lower()]

    # Use raw .X as fallback
    rna_data = adata.obsm[rna_key[0]] if rna_key else adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    prot_data = adata.obsm[prot_key[0]] if prot_key else np.zeros((n_cells, 50))

    # Sort by pseudotime
    pt = adata.obs["pseudotime"].values if "pseudotime" in adata.obs else adata.obs["dpt_pseudotime"].values
    sort_idx = np.argsort(pt)

    # Build windowed sequences
    X_rna_seqs = []
    X_prot_seqs = []
    y_stages = []

    for i in range(context_window + horizon, n_cells - horizon):
        window_idx = sort_idx[i - context_window : i]
        target_idx = sort_idx[i + horizon]

        rna_seq = rna_data[window_idx].astype(np.float32)
        prot_seq = prot_data[window_idx].astype(np.float32) if prot_data.shape[0] > 0 else np.zeros((context_window, 50), dtype=np.float32)

        X_rna_seqs.append(rna_seq)
        X_prot_seqs.append(prot_seq)
        y_stages.append(lifecycle_labels[target_idx])

    X_rna = np.stack(X_rna_seqs)
    X_prot = np.stack(X_prot_seqs)
    y = np.array(y_stages, dtype=np.int64)
    pt_seq = pt[sort_idx[context_window + horizon : n_cells - horizon]]
    pt_seq = pt_seq[:len(y)]

    # Train/val/test split
    n = len(y)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    data = {
        "X_rna": X_rna, "X_prot": X_prot, "y": y,
        "pseudotime": pt_seq,
        "train_idx": indices[:n_train],
        "val_idx": indices[n_train:n_train + n_val],
        "test_idx": indices[n_train + n_val:],
        "n_classes": len(np.unique(y)),
    }
    return data


def build_model(model_name, rna_dim, prot_dim, n_classes, d_model=128):
    """Build the specified model."""
    if model_name == "mlp":
        return nn.Sequential(
            nn.Linear(rna_dim + prot_dim, d_model * 4),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_model * 2, n_classes)
        )
    elif model_name == "lstm":
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rna_proj = nn.Linear(rna_dim, d_model)
                self.prot_proj = nn.Linear(prot_dim, d_model)
                self.lstm = nn.LSTM(d_model * 2, d_model, batch_first=True, bidirectional=True)
                self.head = nn.Linear(d_model * 2, n_classes)
            def forward(self, x_rna, x_prot):
                h_r = self.rna_proj(x_rna)
                h_p = self.prot_proj(x_prot)
                h = torch.cat([h_r, h_p], dim=-1)
                _, (h_n, _) = self.lstm(h)
                h_out = torch.cat([h_n[-2], h_n[-1]], dim=-1)
                return self.head(h_out)
        return LSTMModel()
    elif model_name == "mamba":
        from mamba_ssm import Mamba
        class MambaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(rna_dim + prot_dim, d_model)
                self.mamba = Mamba(d_model=d_model)
                self.head = nn.Linear(d_model, n_classes)
            def forward(self, x_rna, x_prot):
                h = self.proj(torch.cat([x_rna, x_prot], dim=-1))
                h = self.mamba(h)
                return self.head(h[:, -1, :])
        return MambaModel()
    elif model_name in ("mamba_lstm", "lag_aware_fusion"):
        from mamba_ssm import Mamba
        class MambaLSTMModel(nn.Module):
            def __init__(self, lag_aware=False):
                super().__init__()
                self.rna_proj = nn.Linear(rna_dim, d_model)
                self.prot_proj = nn.Linear(prot_dim, d_model)
                self.mamba = Mamba(d_model=d_model)
                self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
                self.lag_aware = lag_aware
                self.head = nn.Linear(d_model * 2, n_classes)
            def forward(self, x_rna, x_prot, pseudotime=None, horizon=None):
                h_r = self.rna_proj(x_rna)
                h_p = self.prot_proj(x_prot)
                h = h_r + h_p  # simple fusion for non-lag-aware
                h_m = self.mamba(h)
                h_l, _ = self.lstm(h_m)
                h_out = torch.cat([h_l[:, -1, :d_model//2], h_l[:, -1, d_model//2:]], dim=-1)
                return self.head(h_out)
        return MambaLSTMModel(lag_aware=(model_name == "lag_aware_fusion"))

    raise ValueError(f"Unknown model: {model_name}")


def train_and_eval(model_name, data, horizon, seed, context_window, device="cuda", epochs=50):
    """Train and evaluate a model."""
    model_dir = get_model_dir(model_name, horizon, seed, context_window)

    # Skip if completed
    status_path = os.path.join(model_dir, "run_status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            st = json.load(f)
        if st.get("status") == "completed":
            print(f"  SKIP: {model_name} h{horizon} s{seed} — already completed")
            return

    torch.manual_seed(seed)
    np.random.seed(seed)

    B, L = data["X_rna"].shape[0], data["X_rna"].shape[1]
    rna_dim = data["X_rna"].shape[2]
    prot_dim = data["X_prot"].shape[2] if data["X_prot"].ndim == 3 else 50

    model = build_model(model_name, rna_dim, prot_dim, data["n_classes"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Data loaders
    X_rna = torch.FloatTensor(data["X_rna"])
    X_prot = torch.FloatTensor(data["X_prot"]) if data["X_prot"].shape[0] > 0 else torch.zeros(B, L, 50)
    y = torch.LongTensor(data["y"])

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_rna[data["train_idx"]], X_prot[data["train_idx"]], y[data["train_idx"]]),
        batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_rna[data["val_idx"]], X_prot[data["val_idx"]], y[data["val_idx"]]),
        batch_size=64, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_rna[data["test_idx"]], X_prot[data["test_idx"]], y[data["test_idx"]]),
        batch_size=64, shuffle=False
    )

    history = []
    best_val_acc = 0

    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_xr, batch_xp, batch_y in train_loader:
                batch_xr, batch_xp, batch_y = batch_xr.to(device), batch_xp.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_xr, batch_xp)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            correct, total_v = 0, 0
            with torch.no_grad():
                for batch_xr, batch_xp, batch_y in val_loader:
                    batch_xr, batch_xp, batch_y = batch_xr.to(device), batch_xp.to(device), batch_y.to(device)
                    logits = model(batch_xr, batch_xp)
                    pred = logits.argmax(dim=1)
                    correct += (pred == batch_y).sum().item()
                    total_v += batch_y.size(0)
            val_acc = correct / total_v
            history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader), "val_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))

        # Final test evaluation
        model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
        model.eval()

        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_xr, batch_xp, batch_y in test_loader:
                batch_xr, batch_xp = batch_xr.to(device), batch_xp.to(device)
                logits = model(batch_xr, batch_xp)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(batch_y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        per_stage_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
        cm = confusion_matrix(all_labels, all_preds).tolist()

        metrics = {
            "accuracy": float(acc),
            "macro_f1": float(f1_macro),
            "balanced_accuracy": float(bal_acc),
            "per_stage_f1": per_stage_f1,
            "confusion_matrix": cm,
            "best_val_acc": float(best_val_acc),
        }

        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        run_status = {"status": "completed", "model": model_name, "horizon": horizon, "seed": seed,
                       "context_window": context_window, "n_classes": data["n_classes"]}
        with open(os.path.join(model_dir, "run_status.json"), "w") as f:
            json.dump(run_status, f, indent=2)

        print(f"  DONE: {model_name} h{horizon} s{seed} | acc={acc:.4f} f1={f1_macro:.4f}")

    except Exception as e:
        run_status = {"status": "failed", "model": model_name, "horizon": horizon, "seed": seed, "error": str(e)}
        with open(os.path.join(model_dir, "run_status.json"), "w") as f:
            json.dump(run_status, f, indent=2)
        print(f"  FAILED: {model_name} h{horizon} s{seed} | {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=MODELS + ["all"])
    parser.add_argument("--horizon", type=int, default=4, choices=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43, 44])
    parser.add_argument("--context_window", type=int, default=8, choices=[4, 8, 16, 32])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    models_to_run = MODELS if args.model == "all" else [args.model]
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"  Lifecycle Prediction: {len(models_to_run)} models")
    print(f"  Horizon: {args.horizon}, Seed: {args.seed}, Context: {args.context_window}")
    print("=" * 60)

    data = load_data(args.context_window, args.horizon)
    print(f"  Data: {data['X_rna'].shape}, Classes: {data['n_classes']}")

    for model_name in models_to_run:
        train_and_eval(model_name, data, args.horizon, args.seed, args.context_window, device, args.epochs)

    return 0

if __name__ == "__main__":
    sys.exit(main())
