#!/usr/bin/env python3
"""
104_run_trajectory_direction_prediction.py
Trajectory direction prediction: classify forward/stationary/backward movement.
Uses pseudotime deltas within the trajectory window to assign direction labels.

Usage:
  python code/scripts/104_run_trajectory_direction_prediction.py --model mamba_lstm --horizon 4 --seed 42
  python code/scripts/104_run_trajectory_direction_prediction.py --model all --horizon 4 --seed 42
"""
import os, sys, json
from pathlib import Path, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "direction_prediction")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["mlp", "lstm", "transformer", "mamba", "mamba_lstm", "lag_aware_fusion"]


def get_model_dir(model_name, horizon, seed, context_window):
    d = os.path.join(OUT_DIR, model_name, f"h{horizon}_s{seed}_ctx{context_window}")
    os.makedirs(d, exist_ok=True)
    return d


def build_direction_labels(pseudotime, delta_threshold=0.01):
    """Build direction labels from pseudotime trajectories.
    Compares pt[-1] vs pt[0] of each window to assign forward/stationary/backward.
    Returns labels: 0=forward, 1=stationary, 2=backward
    """
    N, L = pseudotime.shape
    pt_delta = pseudotime[:, -1] - pseudotime[:, 0]

    labels = np.zeros(N, dtype=np.int64)
    labels[pt_delta > delta_threshold] = 0   # forward
    labels[np.abs(pt_delta) <= delta_threshold] = 1  # stationary
    labels[pt_delta < -delta_threshold] = 2  # backward

    return labels


def load_data(context_window=8, horizon=4):
    pt_path = os.path.join(PROJECT_ROOT,
        "outputs/highdim_real_mamba/sequences/pbmc_citeseq/rna_hvg1000_protein_concat/pseudotime_window_L32")
    seq_path = pt_path

    pseudotime = np.load(os.path.join(pt_path, "pseudotime.npy"))
    sequences = np.load(os.path.join(seq_path, "sequences.npy"))

    N, L, D = sequences.shape
    rna_dim = 1000
    prot_dim = D - rna_dim

    X_rna = sequences[:, :, :rna_dim].astype(np.float32)
    X_prot = sequences[:, :, rna_dim:].astype(np.float32)

    y_direction = build_direction_labels(pseudotime)

    n = len(y_direction)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    return {
        "X_rna": X_rna, "X_prot": X_prot, "y": y_direction,
        "train_idx": indices[:n_train],
        "val_idx": indices[n_train:n_train + n_val],
        "test_idx": indices[n_train + n_val:],
        "n_classes": 3,
    }


def build_model(model_name, rna_dim, prot_dim, n_classes, d_model=128):
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
                h = torch.cat([self.rna_proj(x_rna), self.prot_proj(x_prot)], dim=-1)
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
    elif model_name == "transformer":
        class TransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(rna_dim + prot_dim, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
                self.head = nn.Linear(d_model, n_classes)
            def forward(self, x_rna, x_prot):
                h = self.proj(torch.cat([x_rna, x_prot], dim=-1))
                h = self.transformer(h)
                return self.head(h[:, -1, :])
        return TransformerModel()
    elif model_name in ("mamba_lstm", "lag_aware_fusion"):
        from mamba_ssm import Mamba
        class MambaLSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rna_proj = nn.Linear(rna_dim, d_model)
                self.prot_proj = nn.Linear(prot_dim, d_model)
                self.mamba = Mamba(d_model=d_model)
                self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
                self.head = nn.Linear(d_model * 2, n_classes)
            def forward(self, x_rna, x_prot):
                h = self.rna_proj(x_rna) + self.prot_proj(x_prot)
                h_m = self.mamba(h)
                h_l, _ = self.lstm(h_m)
                h_out = torch.cat([h_l[:, -1, :d_model//2], h_l[:, -1, d_model//2:]], dim=-1)
                return self.head(h_out)
        return MambaLSTMModel()

    raise ValueError(f"Unknown model: {model_name}")


def train_and_eval(model_name, data, horizon, seed, context_window, device="cuda", epochs=50):
    model_dir = get_model_dir(model_name, horizon, seed, context_window)

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
    prot_dim = data["X_prot"].shape[2]

    model = build_model(model_name, rna_dim, prot_dim, data["n_classes"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_rna = torch.FloatTensor(data["X_rna"])
    X_prot = torch.FloatTensor(data["X_prot"])
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

            model.eval()
            correct, total_v = 0, 0
            with torch.no_grad():
                for batch_xr, batch_xp, batch_y in val_loader:
                    batch_xr, batch_xp, batch_y = batch_xr.to(device), batch_xp.to(device), batch_y.to(device)
                    logits = model(batch_xr, batch_xp)
                    correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    total_v += batch_y.size(0)
            val_acc = correct / total_v

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))

        # Test
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
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
        cm = confusion_matrix(all_labels, all_preds).tolist()

        metrics = {
            "accuracy": float(acc), "macro_f1": float(f1_macro),
            "balanced_accuracy": float(bal_acc),
            "per_class_f1": {"forward": per_class_f1[0], "stationary": per_class_f1[1], "backward": per_class_f1[2]},
            "confusion_matrix": cm, "best_val_acc": float(best_val_acc),
        }

        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        run_status = {"status": "completed", "model": model_name, "horizon": horizon,
                       "seed": seed, "context_window": context_window}
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
    print(f"  Direction Prediction: {len(models_to_run)} models")
    print(f"  Horizon: {args.horizon}, Seed: {args.seed}, Context: {args.context_window}")
    print("=" * 60)

    data = load_data(args.context_window, args.horizon)
    print(f"  Data: {data['X_rna'].shape}, Classes: {data['n_classes']}")
    unique, counts = np.unique(data["y"], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {u}: {c} ({c/len(data['y']):.2%})")

    for model_name in models_to_run:
        train_and_eval(model_name, data, args.horizon, args.seed, args.context_window, device, args.epochs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
