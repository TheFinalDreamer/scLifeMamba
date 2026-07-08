#!/usr/bin/env python3
"""
105_run_ablation_study.py
8-mode ablation study for LagAwareDynamicFusion.
Isolates contribution of pseudotime gating, horizon embedding, and fusion strategy.

Usage:
  python code/scripts/105_run_ablation_study.py --mode all --horizon 4 --seed 42
  python code/scripts/105_run_ablation_study.py --mode pt_only --horizon 4 --seed 42
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

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "ablation")
os.makedirs(OUT_DIR, exist_ok=True)

ABLATION_MODES = [
    "concat_fusion",
    "static_average",
    "static_gated",
    "pt_only",
    "horizon_only",
    "full_pt_horizon",
    "full_task_embed",
    "rna_only",
    "protein_only",
]


def get_model_dir(mode, horizon, seed, context_window):
    d = os.path.join(OUT_DIR, mode, f"h{horizon}_s{seed}_ctx{context_window}")
    os.makedirs(d, exist_ok=True)
    return d


def load_data(context_window=8, horizon=4):
    pt_path = os.path.join(PROJECT_ROOT,
        "outputs/highdim_real_mamba/sequences/pbmc_citeseq/rna_hvg1000_protein_concat/pseudotime_window_L32")

    pseudotime = np.load(os.path.join(pt_path, "pseudotime.npy"))
    sequences = np.load(os.path.join(pt_path, "sequences.npy"))

    N, L, D = sequences.shape
    rna_dim = 1000
    prot_dim = D - rna_dim

    X_rna = sequences[:, :, :rna_dim].astype(np.float32)
    X_prot = sequences[:, :, rna_dim:].astype(np.float32)

    # Lifecycle labels from pseudotime quantiles
    pt_flat = pseudotime.ravel()
    pt_norm = (pt_flat - pt_flat.min()) / (pt_flat.max() - pt_flat.min())
    boundaries = np.quantile(pt_norm, np.linspace(0, 1, 5))
    labels_flat = np.digitize(pt_norm, boundaries[1:-1])
    labels_flat = np.clip(labels_flat, 0, 3)
    y_stage = labels_flat.reshape(pseudotime.shape)[:, -1]

    n = len(y_stage)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    return {
        "X_rna": X_rna, "X_prot": X_prot, "y": y_stage,
        "pseudotime_mean": pseudotime.mean(axis=1).astype(np.float32),
        "train_idx": indices[:n_train],
        "val_idx": indices[n_train:n_train + n_val],
        "test_idx": indices[n_train + n_val:],
        "n_classes": 4,
    }


def build_ablation_model(mode, rna_dim, prot_dim, n_classes, d_model=128):
    """Build model for a specific ablation mode."""
    from mamba_ssm import Mamba

    class AblationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.rna_proj = nn.Linear(rna_dim, d_model)
            self.prot_proj = nn.Linear(prot_dim, d_model)
            self.mamba = Mamba(d_model=d_model)
            self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)

            # Fusion components
            embed_dim = d_model // 4
            if "pt" in mode or mode in ("full_pt_horizon", "full_task_embed"):
                self.pt_embed = nn.Embedding(20, embed_dim)
            if "horizon" in mode or mode in ("full_pt_horizon", "full_task_embed"):
                self.horizon_embed = nn.Embedding(17, embed_dim)

            # Gate network for gated modes
            gate_input_dim = self._get_gate_input_dim()
            if gate_input_dim > 0:
                self.gate = nn.Sequential(
                    nn.Linear(gate_input_dim, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, 2),
                )
            else:
                self.gate = None

            self.head = nn.Linear(d_model * 2, n_classes)

        def _get_gate_input_dim(self):
            if self.mode == "static_gated":
                return d_model * 2
            elif self.mode == "pt_only":
                return d_model // 4
            elif self.mode == "horizon_only":
                return d_model // 4
            elif self.mode in ("full_pt_horizon", "full_task_embed"):
                return d_model * 2 + d_model // 2
            return 0

        def forward(self, x_rna, x_prot, pseudotime=None, horizon=None):
            h_r = self.rna_proj(x_rna)
            h_p = self.prot_proj(x_prot)

            # Encode with Mamba
            if self.mode in ("rna_only",):
                h = h_r
            elif self.mode in ("protein_only",):
                h = h_p
            else:
                h = h_r + h_p

            h_m = self.mamba(h)
            h_l, _ = self.lstm(h_m)
            z = torch.cat([h_l[:, -1, :d_model//2], h_l[:, -1, d_model//2:]], dim=-1)

            # Fusion strategy
            h_r_last = h_r[:, -1, :]
            h_p_last = h_p[:, -1, :]

            if self.mode == "concat_fusion":
                z_fused = torch.cat([h_r_last, h_p_last], dim=-1)
            elif self.mode == "static_average":
                z_fused = (h_r_last + h_p_last) / 2
            elif self.mode == "static_gated":
                gate_input = torch.cat([h_r_last, h_p_last], dim=-1)
                alpha = torch.softmax(self.gate(gate_input), dim=-1)
                z_fused = alpha[:, 0:1] * h_r_last + alpha[:, 1:2] * h_p_last
            elif self.mode == "pt_only":
                pt_bin = (pseudotime * 19).long().clamp(0, 19)
                e_pt = self.pt_embed(pt_bin)
                alpha = torch.softmax(self.gate(e_pt), dim=-1)
                z_fused = alpha[:, 0:1] * h_r_last + alpha[:, 1:2] * h_p_last
            elif self.mode == "horizon_only":
                h_bin = torch.full((x_rna.size(0),), min(horizon, 16), device=x_rna.device, dtype=torch.long)
                e_h = self.horizon_embed(h_bin)
                alpha = torch.softmax(self.gate(e_h), dim=-1)
                z_fused = alpha[:, 0:1] * h_r_last + alpha[:, 1:2] * h_p_last
            elif self.mode == "full_pt_horizon":
                pt_bin = (pseudotime * 19).long().clamp(0, 19)
                h_bin = torch.full((x_rna.size(0),), min(horizon, 16), device=x_rna.device, dtype=torch.long)
                gate_input = torch.cat([h_r_last, h_p_last, self.pt_embed(pt_bin), self.horizon_embed(h_bin)], dim=-1)
                alpha = torch.softmax(self.gate(gate_input), dim=-1)
                z_fused = alpha[:, 0:1] * h_r_last + alpha[:, 1:2] * h_p_last
            elif self.mode == "full_task_embed":
                pt_bin = (pseudotime * 19).long().clamp(0, 19)
                h_bin = torch.full((x_rna.size(0),), min(horizon, 16), device=x_rna.device, dtype=torch.long)
                gate_input = torch.cat([h_r_last, h_p_last, self.pt_embed(pt_bin), self.horizon_embed(h_bin)], dim=-1)
                alpha = torch.softmax(self.gate(gate_input), dim=-1)
                z_fused = alpha[:, 0:1] * h_r_last + alpha[:, 1:2] * h_p_last
            elif self.mode in ("rna_only", "protein_only"):
                z_fused = z
            else:
                z_fused = z

            if self.mode in ("rna_only", "protein_only"):
                return self.head(z_fused)
            else:
                # For fusion modes, combine z (sequence context) with fused modality representation
                return self.head(z_fused if self.mode == "concat_fusion" else
                    torch.cat([h_r_last, h_p_last], dim=-1) if self.mode == "static_average" else
                    z_fused if z_fused.shape[-1] == d_model * 2 else z)

    return AblationModel()


def train_and_eval(mode, data, horizon, seed, context_window, device="cuda", epochs=50):
    model_dir = get_model_dir(mode, horizon, seed, context_window)

    status_path = os.path.join(model_dir, "run_status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            st = json.load(f)
        if st.get("status") == "completed":
            print(f"  SKIP: {mode} h{horizon} s{seed} — already completed")
            return

    torch.manual_seed(seed)
    np.random.seed(seed)

    rna_dim = data["X_rna"].shape[2]
    prot_dim = data["X_prot"].shape[2]

    model = build_ablation_model(mode, rna_dim, prot_dim, data["n_classes"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_rna = torch.FloatTensor(data["X_rna"])
    X_prot = torch.FloatTensor(data["X_prot"])
    y = torch.LongTensor(data["y"])
    pseudotime = torch.FloatTensor(data["pseudotime_mean"])

    train_ds = torch.utils.data.TensorDataset(
        X_rna[data["train_idx"]], X_prot[data["train_idx"]],
        y[data["train_idx"]], pseudotime[data["train_idx"]]
    )
    val_ds = torch.utils.data.TensorDataset(
        X_rna[data["val_idx"]], X_prot[data["val_idx"]],
        y[data["val_idx"]], pseudotime[data["val_idx"]]
    )
    test_ds = torch.utils.data.TensorDataset(
        X_rna[data["test_idx"]], X_prot[data["test_idx"]],
        y[data["test_idx"]], pseudotime[data["test_idx"]]
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    best_val_acc = 0

    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_xr, batch_xp, batch_y, batch_pt in train_loader:
                batch_xr, batch_xp, batch_y, batch_pt = (
                    batch_xr.to(device), batch_xp.to(device), batch_y.to(device), batch_pt.to(device)
                )
                optimizer.zero_grad()
                logits = model(batch_xr, batch_xp, batch_pt, horizon)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            model.eval()
            correct, total_v = 0, 0
            with torch.no_grad():
                for batch_xr, batch_xp, batch_y, batch_pt in val_loader:
                    batch_xr, batch_xp, batch_y, batch_pt = (
                        batch_xr.to(device), batch_xp.to(device), batch_y.to(device), batch_pt.to(device)
                    )
                    logits = model(batch_xr, batch_xp, batch_pt, horizon)
                    correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    total_v += batch_y.size(0)
            val_acc = correct / total_v

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))

        # Test
        model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
        model.eval()

        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_xr, batch_xp, batch_y, batch_pt in test_loader:
                batch_xr, batch_xp, batch_pt = batch_xr.to(device), batch_xp.to(device), batch_pt.to(device)
                logits = model(batch_xr, batch_xp, batch_pt, horizon)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(batch_y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)

        metrics = {
            "accuracy": float(acc), "macro_f1": float(f1_macro),
            "balanced_accuracy": float(bal_acc), "best_val_acc": float(best_val_acc),
        }

        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        run_status = {"status": "completed", "mode": mode, "horizon": horizon,
                       "seed": seed, "context_window": context_window}
        with open(os.path.join(model_dir, "run_status.json"), "w") as f:
            json.dump(run_status, f, indent=2)

        print(f"  DONE: {mode} h{horizon} s{seed} | acc={acc:.4f} f1={f1_macro:.4f}")

    except Exception as e:
        run_status = {"status": "failed", "mode": mode, "horizon": horizon, "seed": seed, "error": str(e)}
        with open(os.path.join(model_dir, "run_status.json"), "w") as f:
            json.dump(run_status, f, indent=2)
        print(f"  FAILED: {mode} h{horizon} s{seed} | {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=ABLATION_MODES + ["all"])
    parser.add_argument("--horizon", type=int, default=4, choices=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43, 44])
    parser.add_argument("--context_window", type=int, default=8, choices=[4, 8, 16, 32])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    modes_to_run = ABLATION_MODES if args.mode == "all" else [args.mode]
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"  Ablation Study: {len(modes_to_run)} modes")
    print(f"  Horizon: {args.horizon}, Seed: {args.seed}, Context: {args.context_window}")
    print("=" * 60)

    data = load_data(args.context_window, args.horizon)
    print(f"  Data: {data['X_rna'].shape}, Classes: {data['n_classes']}")

    for mode in modes_to_run:
        train_and_eval(mode, data, args.horizon, args.seed, args.context_window, device, args.epochs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
