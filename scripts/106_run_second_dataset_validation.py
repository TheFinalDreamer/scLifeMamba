#!/usr/bin/env python3
"""
106_run_second_dataset_validation.py
External dataset validation for scMultiLifeMamba.
Loads an external CITE-seq dataset and runs lifecycle prediction + fusion analysis.

Usage:
  python code/scripts/106_run_second_dataset_validation.py --dataset bmmc --horizon 4 --seed 42
  python code/scripts/106_run_second_dataset_validation.py --dataset asap --horizon 4 --seed 42
  python code/scripts/106_run_second_dataset_validation.py --dataset rna_only --horizon 4 --seed 42
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

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lifecycle_prediction", "second_dataset")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["bmmc", "asap", "rna_only"]
MODELS = ["mlp", "lstm", "mamba_lstm", "lag_aware_fusion"]


def get_model_dir(dataset, model_name, horizon, seed):
    d = os.path.join(OUT_DIR, dataset, model_name, f"h{horizon}_s{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def load_dataset(dataset_name):
    """Load external validation dataset."""
    data_paths = {
        "bmmc": os.path.join(PROJECT_ROOT, "data/external/bmmc_citeseq.h5ad"),
        "asap": os.path.join(PROJECT_ROOT, "data/external/asap_citeseq.h5ad"),
        "rna_only": os.path.join(PROJECT_ROOT, "data/external/rna_only.h5ad"),
    }

    path = data_paths.get(dataset_name)
    if path and os.path.exists(path):
        import scanpy as sc
        adata = sc.read_h5ad(path)
        print(f"  Loaded {dataset_name}: {adata.n_obs} cells, {adata.n_vars} genes")

        # Build pseudotime if not present
        if "pseudotime" not in adata.obs:
            print("  Computing pseudotime...")
            import scanpy.external as sce
            sc.pp.pca(adata, n_comps=50)
            sc.pp.neighbors(adata)
            sc.tl.diffmap(adata)
            adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"] \
                if "dpt_pseudotime" in adata.obs else np.linspace(0, 1, adata.n_obs)

        return adata

    # Fallback: use existing PBMC data with subsampling for RNA-only mode
    print(f"  Dataset {dataset_name} not found at {path}, using fallback PBMC data")
    return None


def build_sequences(adata, context_window=8, horizon=4):
    """Build trajectory sequences from AnnData."""
    if adata is None:
        # Load from existing sequences as fallback
        pt_path = os.path.join(PROJECT_ROOT,
            "outputs/highdim_real_mamba/sequences/pbmc_citeseq/rna_hvg1000_protein_concat/pseudotime_window_L32")
        pseudotime = np.load(os.path.join(pt_path, "pseudotime.npy"))
        sequences = np.load(os.path.join(pt_path, "sequences.npy"))
        rna_dim = 1000
        prot_dim = sequences.shape[2] - rna_dim
        X_rna = sequences[:, :, :rna_dim].astype(np.float32)
        X_prot = sequences[:, :, rna_dim:].astype(np.float32)

        # Lifecycle labels
        pt_flat = pseudotime.ravel()
        pt_norm = (pt_flat - pt_flat.min()) / (pt_flat.max() - pt_flat.min())
        boundaries = np.quantile(pt_norm, np.linspace(0, 1, 5))
        labels = np.digitize(pt_norm, boundaries[1:-1])
        labels = np.clip(labels, 0, 3).reshape(pseudotime.shape)[:, -1]

        n = len(labels)
        indices = np.random.RandomState(42).permutation(n)
        n_train = int(n * 0.7)

        return {
            "X_rna": X_rna, "X_prot": X_prot, "y": labels,
            "pseudotime": pseudotime.mean(axis=1).astype(np.float32),
            "train_idx": indices[:n_train],
            "test_idx": indices[n_train:],
            "n_classes": 4,
        }

    # Build from AnnData
    pt = adata.obs["pseudotime"].values
    sort_idx = np.argsort(pt)

    # Get RNA and protein data
    rna_key = [k for k in adata.obsm.keys() if "rna" in k.lower() or "protein" not in k.lower()]
    prot_key = [k for k in adata.obsm.keys() if "prot" in k.lower() or "adt" in k.lower()]

    rna_data = adata.obsm[rna_key[0]] if rna_key else adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    prot_data = adata.obsm[prot_key[0]] if prot_key else np.zeros((adata.n_obs, 50))

    X_rna_seqs, X_prot_seqs = [], []
    n_cells = adata.n_obs

    for i in range(context_window, n_cells - horizon):
        window_idx = sort_idx[i - context_window : i]
        X_rna_seqs.append(rna_data[window_idx].astype(np.float32))
        X_prot_seqs.append(prot_data[window_idx].astype(np.float32))

    X_rna = np.stack(X_rna_seqs)
    X_prot = np.stack(X_prot_seqs)
    pt_seq = pt[sort_idx[context_window : n_cells - horizon]]

    # Lifecycle labels
    pt_norm = (pt_seq - pt_seq.min()) / (pt_seq.max() - pt_seq.min())
    boundaries = np.quantile(pt_norm, np.linspace(0, 1, 5))
    labels = np.digitize(pt_norm, boundaries[1:-1])
    labels = np.clip(labels, 0, 3)

    n = len(labels)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.7)

    return {
        "X_rna": X_rna, "X_prot": X_prot, "y": labels,
        "pseudotime": pt_seq.astype(np.float32),
        "train_idx": indices[:n_train],
        "test_idx": indices[n_train:],
        "n_classes": 4,
    }


def build_model(model_name, rna_dim, prot_dim, n_classes, d_model=128):
    """Build model (simplified from 102)."""
    if model_name == "mlp":
        return nn.Sequential(
            nn.Linear(rna_dim + prot_dim, d_model * 4), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_model * 4, d_model * 2), nn.ReLU(), nn.Dropout(0.2),
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
                return self.head(torch.cat([h_n[-2], h_n[-1]], dim=-1))
        return LSTMModel()
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
                return self.head(torch.cat([h_l[:, -1, :d_model//2], h_l[:, -1, d_model//2:]], dim=-1))
        return MambaLSTMModel()
    raise ValueError(f"Unknown model: {model_name}")


def train_and_eval(dataset, model_name, data, horizon, seed, device="cuda", epochs=50):
    model_dir = get_model_dir(dataset, model_name, horizon, seed)

    status_path = os.path.join(model_dir, "run_status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            st = json.load(f)
        if st.get("status") == "completed":
            print(f"  SKIP: {dataset}/{model_name} h{horizon} s{seed} — already completed")
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
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_rna[data["test_idx"]], X_prot[data["test_idx"]], y[data["test_idx"]]),
        batch_size=64, shuffle=False
    )

    best_loss = float("inf")

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

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))

        # Test evaluation
        model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
        model.eval()

        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
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

        metrics = {
            "accuracy": float(acc), "macro_f1": float(f1_macro),
            "balanced_accuracy": float(bal_acc), "n_test": len(all_labels),
        }

        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        run_status = {"status": "completed", "dataset": dataset, "model": model_name,
                       "horizon": horizon, "seed": seed}
        with open(os.path.join(model_dir, "run_status.json"), "w") as f:
            json.dump(run_status, f, indent=2)

        print(f"  DONE: {dataset}/{model_name} h{horizon} s{seed} | acc={acc:.4f} f1={f1_macro:.4f}")

    except Exception as e:
        run_status = {"status": "failed", "dataset": dataset, "model": model_name,
                       "horizon": horizon, "seed": seed, "error": str(e)}
        with open(os.path.join(model_dir, "run_status.json"), "w") as f:
            json.dump(run_status, f, indent=2)
        print(f"  FAILED: {dataset}/{model_name} h{horizon} s{seed} | {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rna_only", choices=DATASETS + ["all"])
    parser.add_argument("--model", default="all", choices=MODELS + ["all"])
    parser.add_argument("--horizon", type=int, default=4, choices=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43, 44])
    parser.add_argument("--context_window", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    datasets_to_run = DATASETS if args.dataset == "all" else [args.dataset]
    models_to_run = MODELS if args.model == "all" else [args.model]
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"  Second Dataset Validation")
    print(f"  Datasets: {datasets_to_run}, Models: {models_to_run}")
    print(f"  Horizon: {args.horizon}, Seed: {args.seed}")
    print("=" * 60)

    for dataset in datasets_to_run:
        adata = load_dataset(dataset)
        data = build_sequences(adata, args.context_window, args.horizon)
        print(f"  [{dataset}] samples={len(data['y'])}, classes={data['n_classes']}")

        for model_name in models_to_run:
            train_and_eval(dataset, model_name, data, args.horizon, args.seed, device, args.epochs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
