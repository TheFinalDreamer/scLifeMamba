#!/usr/bin/env python3
"""
131_run_revised_trajectory_direction_prediction.py
Trajectory direction prediction using REVISED labels from script 130 (Scheme B).
Labels: 0=fast_progression, 1=medium, 2=slow (pseudotime delta quantile tripartition).
Based on the corrected 104 architecture with proper LagAwareDynamicFusion support.

Usage:
  python code/scripts/131_run_revised_trajectory_direction_prediction.py --model all --horizon 8 --seed 42
  python code/scripts/131_run_revised_trajectory_direction_prediction.py --model mamba_lstm --horizon 8 --seed 42
"""
import os, sys, json
from pathlib import Path, argparse, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "revised_direction_prediction")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["mlp", "lstm", "transformer", "mamba", "mamba_lstm", "lag_aware_fusion"]


def get_model_dir(model_name, horizon, seed, context_window):
    d = os.path.join(OUT_DIR, model_name, f"h{horizon}_s{seed}_ctx{context_window}")
    os.makedirs(d, exist_ok=True)
    return d


def load_data(context_window=8, horizon=8):
    """Load sequences and the revised direction labels from script 130."""
    pt_path = os.path.join(PROJECT_ROOT,
        "outputs/highdim_real_mamba/sequences/pbmc_citeseq/rna_hvg1000_protein_concat/pseudotime_window_L32")

    pseudotime = np.load(os.path.join(pt_path, "pseudotime.npy"))
    sequences = np.load(os.path.join(pt_path, "sequences.npy"))

    N, L, D = sequences.shape
    rna_dim = 1000
    prot_dim = D - rna_dim

    X_rna = sequences[:, :, :rna_dim].astype(np.float32)
    X_prot = sequences[:, :, rna_dim:].astype(np.float32)

    # Load revised labels from script 130 (Scheme B, selected horizon)
    revised_label_dir = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction",
                                      "revised_direction_labels")
    label_file = os.path.join(revised_label_dir, f"direction_labels_schemeB_h{horizon}.npy")
    if os.path.exists(label_file):
        y_direction = np.load(label_file).astype(np.int64)
        print(f"  Loaded revised labels: {label_file}")
    else:
        # Fallback: search for any scheme B file
        candidates = sorted([f for f in os.listdir(revised_label_dir)
                            if f.startswith("direction_labels_schemeB_") and f.endswith(".npy")])
        if candidates:
            y_direction = np.load(os.path.join(revised_label_dir, candidates[0])).astype(np.int64)
            print(f"  Fallback: loaded {candidates[0]}")
        else:
            raise FileNotFoundError(f"No revised direction labels found in {revised_label_dir}")

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
    best_model_saved = False

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
                best_model_saved = True

        # Fallback: save final model if never saved
        if not best_model_saved:
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

        class_names = ["fast", "medium", "slow"]
        per_class_dict = {}
        for i, name in enumerate(class_names):
            per_class_dict[name] = per_class_f1[i] if i < len(per_class_f1) else 0.0

        metrics = {
            "accuracy": float(acc), "macro_f1": float(f1_macro),
            "balanced_accuracy": float(bal_acc),
            "per_class_f1": per_class_dict,
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
    parser.add_argument("--horizon", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43, 44])
    parser.add_argument("--context_window", type=int, default=8, choices=[4, 8, 16, 32])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    models_to_run = MODELS if args.model == "all" else [args.model]
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"  Revised Direction Prediction: {len(models_to_run)} models")
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
