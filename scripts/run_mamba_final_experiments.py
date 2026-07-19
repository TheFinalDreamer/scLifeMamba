"""
Unified experiment runner for Mamba-LSTM method paper (Experiments 1-4).

Usage:
    # Experiment 1: Backbone comparison
    python run_mamba_final_experiments.py --exp backbone --model mlp --horizon 1 --seed 42

    # Experiment 2: Architecture ablation
    python run_mamba_final_experiments.py --exp ablation --model mamba_lstm --ablation full --horizon 1 --seed 42

    # Experiment 3: Sequence dependency
    python run_mamba_final_experiments.py --exp sequence --model mamba_lstm --shuffle random --horizon 1 --seed 42

    # Experiment 4: Generalization (donor-held-out, default)
    python run_mamba_final_experiments.py --exp generalization --model mamba_lstm --horizon 1 --seed 42

    # Run all experiments for a model
    python run_mamba_final_experiments.py --exp all --model mamba_lstm --horizon 1 --seed 42

Output structure:
    outputs/mamba_final/
        {experiment}/{model}/{seed}/{horizon}/
            config.yaml
            metrics.json
            predictions.csv
            training_log.txt
"""

import os
import sys
import json
import yaml
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, classification_report, confusion_matrix,
)

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(PROJECT_ROOT / 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.sequence_dataloader import SequenceDataLoader
from models.scLifeMamba import scLifeMamba
from models.mamba_block import MambaBlock

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────────────────────────────────
# Model Factory
# ────────────────────────────────────────────────────────────────────────────

def build_model(model_type, rna_dim, protein_dim, num_classes, device='cuda', **kwargs):
    """Build model by type.

    Args:
        model_type: 'mlp' | 'transformer' | 'lstm' | 'mamba' | 'mamba_lstm'
    """
    hidden_dim = kwargs.get('hidden_dim', 128)
    dropout = kwargs.get('dropout', 0.2)

    if model_type == 'mlp':
        return scLifeMamba(
            rna_dim=rna_dim, protein_dim=protein_dim, num_classes=num_classes,
            hidden_dim=hidden_dim, dropout=dropout,
            use_mamba=False, use_lstm=False, use_protein=True,
        ).to(device)

    elif model_type == 'transformer':
        return TransformerSeqModel(
            rna_dim=rna_dim, protein_dim=protein_dim, num_classes=num_classes,
            hidden_dim=hidden_dim, dropout=dropout,
        ).to(device)

    elif model_type == 'lstm':
        return scLifeMamba(
            rna_dim=rna_dim, protein_dim=protein_dim, num_classes=num_classes,
            hidden_dim=hidden_dim, dropout=dropout,
            use_mamba=False, use_lstm=True, use_protein=True,
        ).to(device)

    elif model_type == 'mamba':
        return scLifeMamba(
            rna_dim=rna_dim, protein_dim=protein_dim, num_classes=num_classes,
            hidden_dim=hidden_dim, dropout=dropout,
            use_mamba=True, use_lstm=False, use_protein=True,
        ).to(device)

    elif model_type == 'mamba_lstm':
        return scLifeMamba(
            rna_dim=rna_dim, protein_dim=protein_dim, num_classes=num_classes,
            hidden_dim=hidden_dim, dropout=dropout,
            use_mamba=True, use_lstm=True, use_protein=True,
        ).to(device)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class TransformerSeqModel(nn.Module):
    """Simple Transformer encoder for sequence classification (comparison baseline)."""

    def __init__(self, rna_dim, protein_dim, num_classes, hidden_dim=128,
                 nhead=4, nlayers=2, dropout=0.2):
        super().__init__()
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
        )
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fusion_proj = nn.Linear(256 + 64, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=64, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x_rna, x_protein=None):
        B, L, Dr = x_rna.shape
        # Per-position encoding
        z_rna = self.rna_encoder(x_rna.reshape(B * L, Dr)).reshape(B, L, -1)
        if x_protein is not None:
            Dp = x_protein.shape[-1]
            z_prot = self.protein_encoder(x_protein.reshape(B * L, Dp)).reshape(B, L, -1)
            z = self.fusion_proj(torch.cat([z_rna, z_prot], dim=-1))
        else:
            z = self.fusion_proj(torch.cat([z_rna, torch.zeros_like(z_rna[:, :, :64])], dim=-1))

        z = self.pos_encoding(z)
        h = self.transformer(z)
        h_pooled = h.mean(dim=1)
        logits = self.classifier(h_pooled)
        return {'logits': logits}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        x_rna = batch['x_rna'].to(device)
        x_protein = batch['x_protein'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        out = model(x_rna, x_protein)
        loss = criterion(out['logits'], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.append(out['logits'].argmax(-1).detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)

    return {
        'loss': total_loss / len(all_labels),
        'macro_f1': float(macro_f1),
        'accuracy': float(acc),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        x_rna = batch['x_rna'].to(device)
        x_protein = batch['x_protein'].to(device)
        labels = batch['label'].to(device)

        out = model(x_rna, x_protein)
        loss = criterion(out['logits'], labels)

        total_loss += loss.item() * len(labels)
        all_preds.append(out['logits'].argmax(-1).cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': total_loss / len(all_labels),
        'macro_f1': float(f1_score(all_labels, all_preds, average='macro')),
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'balanced_accuracy': float(balanced_accuracy_score(all_labels, all_preds)),
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main Experiment Runner
# ────────────────────────────────────────────────────────────────────────────

def run_single_experiment(config):
    """Run one model training + evaluation. Returns metrics dict."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'training_log.txt'

    def log(msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        with open(log_file, 'a') as f:
            f.write(line + '\n')

    log(f"Starting experiment: {config['experiment']}/{config['model_type']}")
    log(f"  horizon={config['horizon']}, seed={seed}, device={device}")

    # ── Load data ──
    shuffle_order = config.get('shuffle_order', 'original')
    data_dir = config['data_dir']

    try:
        loader = SequenceDataLoader(
            data_dir=data_dir,
            horizon=config['horizon'],
            batch_size=config['batch_size'],
            shuffle_order=shuffle_order,
            seed=seed,
        )
    except Exception as e:
        log(f"ERROR loading data: {e}")
        return {'status': 'error', 'error': str(e)}

    train_loader = loader.train_dataloader()
    val_loader = loader.val_dataloader()
    test_loader = loader.test_dataloader()

    log(f"  Train: {loader.train_size}, Val: {loader.val_size}, Test: {loader.test_size}")

    # ── Build model ──
    model = build_model(
        config['model_type'],
        rna_dim=loader.rna_dim,
        protein_dim=loader.protein_dim,
        num_classes=loader.n_classes,
        device=device,
    )

    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: {config['model_type']}, Params: {n_params:,}")
    if hasattr(model, 'get_mamba_backend'):
        log(f"  Mamba backend: {model.get_mamba_backend()}")

    # ── Training setup ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01,
    )

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = config.get('patience', 10)
    val_every = config.get('val_every', 5)
    train_history = []
    val_history = []

    # ── Training loop ──
    for epoch in range(1, config['epochs'] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        train_history.append(train_metrics)

        # Validate every val_every epochs + first + last
        if epoch % val_every == 0 or epoch == 1 or epoch == config['epochs']:
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_history.append({'epoch': epoch, **val_metrics})

            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
            else:
                patience_counter += 1

            log(f"  Epoch {epoch:3d}: train_loss={train_metrics['loss']:.4f}, "
                f"val_f1={val_metrics['macro_f1']:.4f}, val_acc={val_metrics['accuracy']:.4f}")

            # Early stopping
            if patience_counter >= patience and epoch > 20:
                log(f"  Early stopping at epoch {epoch} (no improvement for {patience} validations)")
                break

    log(f"  Best: epoch={best_epoch}, val_f1={best_val_f1:.4f}")

    # ── Final evaluation ──
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_metrics = evaluate(model, test_loader, criterion, device)

    log(f"  Test: macro_f1={test_metrics['macro_f1']:.4f}, acc={test_metrics['accuracy']:.4f}")

    # ── Save results ──
    results = {
        'config': config,
        'model_type': config['model_type'],
        'experiment': config['experiment'],
        'horizon': config['horizon'],
        'seed': seed,
        'shuffle_order': shuffle_order,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'test_macro_f1': test_metrics['macro_f1'],
        'test_accuracy': test_metrics['accuracy'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'test_per_class_f1': test_metrics['per_class_f1'],
        'test_confusion_matrix': test_metrics['confusion_matrix'],
        'train_history': train_history,
        'val_history': val_history,
        'status': 'success',
    }

    # Add Mamba backend info
    if hasattr(model, 'get_mamba_backend'):
        results['mamba_backend'] = model.get_mamba_backend()

    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': test_metrics['labels'],
        'pred_label': test_metrics['predictions'],
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    log(f"  Results saved to {output_dir}")
    return results


# ────────────────────────────────────────────────────────────────────────────
# Experiment Configuration Builders
# ────────────────────────────────────────────────────────────────────────────

def build_exp1_configs(args):
    """Experiment 1: Backbone comparison.
    Models: mlp, transformer, lstm, mamba, mamba_lstm
    For each: 3 seeds × 4 horizons = 12 runs
    Total: 5 models × 12 = 60 runs
    """
    configs = []
    for model_type in ['mlp', 'transformer', 'lstm', 'mamba', 'mamba_lstm']:
        for horizon in [1, 4, 8, 16]:
            for seed in [42, 123, 456]:
                output_dir = (
                    f"outputs/mamba_final/exp1_backbone/{model_type}/seed{seed}/h{horizon}"
                )
                configs.append({
                    'experiment': 'exp1_backbone',
                    'model_type': model_type,
                    'horizon': horizon,
                    'seed': seed,
                    'shuffle_order': 'original',
                    'output_dir': output_dir,
                    'data_dir': args.data_dir,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'lr': args.lr,
                })
    return configs


def build_exp2_configs(args):
    """Experiment 2: Architecture ablation.
    Models: full, no_mamba, no_lstm, no_rna, no_protein, rna_only, protein_only
    For each: 3 seeds × 4 horizons = 12 runs per variant
    """
    configs = []
    ablations = {
        'full':        {'use_mamba': True,  'use_lstm': True,  'use_protein': True},
        'no_mamba':    {'use_mamba': False, 'use_lstm': True,  'use_protein': True},
        'no_lstm':     {'use_mamba': True,  'use_lstm': False, 'use_protein': True},
        'no_rna':      {'use_mamba': True,  'use_lstm': True,  'use_protein': True},  # protein drives prediction
        'no_protein':  {'use_mamba': True,  'use_lstm': True,  'use_protein': False},
        'rna_only':    {'use_mamba': True,  'use_lstm': True,  'use_protein': False},  # same as no_protein
        'protein_only': {'use_mamba': True, 'use_lstm': True,  'use_protein': True},   # RNA set to zero
    }
    # Deduplicate: rna_only == no_protein
    del ablations['rna_only']
    ablations['protein_only_mode'] = {'use_mamba': True, 'use_lstm': True, 'use_protein': False}

    for ab_name, ab_opts in ablations.items():
        for horizon in [1, 4, 8, 16]:
            for seed in [42, 123, 456]:
                output_dir = (
                    f"outputs/mamba_final/exp2_ablation/{ab_name}/seed{seed}/h{horizon}"
                )
                configs.append({
                    'experiment': 'exp2_ablation',
                    'model_type': 'mamba_lstm',
                    'ablation': ab_name,
                    'horizon': horizon,
                    'seed': seed,
                    'shuffle_order': 'original',
                    'output_dir': output_dir,
                    'data_dir': args.data_dir,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    **ab_opts,
                })
    return configs


def build_exp3_configs(args):
    """Experiment 3: Sequence dependency validation.
    Orders: original, random, no_order
    Model: mamba_lstm
    """
    configs = []
    for shuffle_order in ['original', 'random', 'no_order']:
        for horizon in [1, 4, 8, 16]:
            for seed in [42, 123, 456]:
                output_dir = (
                    f"outputs/mamba_final/exp3_sequence/{shuffle_order}/seed{seed}/h{horizon}"
                )
                configs.append({
                    'experiment': 'exp3_sequence',
                    'model_type': 'mamba_lstm',
                    'horizon': horizon,
                    'seed': seed,
                    'shuffle_order': shuffle_order,
                    'output_dir': output_dir,
                    'data_dir': args.data_dir,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'lr': args.lr,
                })
    return configs


def build_exp4_configs(args):
    """Experiment 4: Generalization (donor-held-out).
    Model: mamba_lstm, default donor split.
    """
    configs = []
    for horizon in [1, 4, 8, 16]:
        for seed in [42, 123, 456]:
            output_dir = (
                f"outputs/mamba_final/exp4_generalization/donor_held_out/seed{seed}/h{horizon}"
            )
            configs.append({
                'experiment': 'exp4_generalization',
                'model_type': 'mamba_lstm',
                'horizon': horizon,
                'seed': seed,
                'shuffle_order': 'original',
                'output_dir': output_dir,
                'data_dir': args.data_dir,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr,
            })
    return configs


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Mamba-LSTM final experiments')
    parser.add_argument('--exp', type=str, default='backbone',
                        choices=['backbone', 'ablation', 'sequence', 'generalization', 'all'])
    parser.add_argument('--model', type=str, default='mamba_lstm',
                        choices=['mlp', 'transformer', 'lstm', 'mamba', 'mamba_lstm'])
    parser.add_argument('--horizon', type=int, default=1,
                        choices=[1, 4, 8, 16])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shuffle', type=str, default='original',
                        choices=['original', 'random', 'no_order'])
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--data_dir', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'leakage_safe_v1'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    # Build configs
    if args.exp == 'backbone':
        configs = build_exp1_configs(args)
        # Filter to match args
        configs = [c for c in configs
                   if c['model_type'] == args.model
                   and c['horizon'] == args.horizon
                   and c['seed'] == args.seed]
    elif args.exp == 'ablation':
        configs = build_exp2_configs(args)
        if args.ablation:
            configs = [c for c in configs if c.get('ablation') == args.ablation]
        configs = [c for c in configs
                   if c['horizon'] == args.horizon
                   and c['seed'] == args.seed]
    elif args.exp == 'sequence':
        configs = build_exp3_configs(args)
        configs = [c for c in configs
                   if c['shuffle_order'] == args.shuffle
                   and c['horizon'] == args.horizon
                   and c['seed'] == args.seed]
    elif args.exp == 'generalization':
        configs = build_exp4_configs(args)
        configs = [c for c in configs
                   if c['horizon'] == args.horizon
                   and c['seed'] == args.seed]
    elif args.exp == 'all':
        configs = (
            build_exp1_configs(args) +
            build_exp2_configs(args) +
            build_exp3_configs(args) +
            build_exp4_configs(args)
        )

    print(f"Experiment: {args.exp}")
    print(f"Configurations to run: {len(configs)}")
    if args.dry_run:
        for c in configs:
            print(f"  {c['experiment']}/{c['model_type']} h={c['horizon']} s={c['seed']} -> {c['output_dir']}")
        return

    # Run experiments
    all_results = []
    t0 = time.time()
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(configs)}: {config['experiment']}/{config['model_type']} "
              f"h={config['horizon']} s={config['seed']}")
        print(f"{'='*60}")

        result = run_single_experiment(config)
        all_results.append(result)

        elapsed = time.time() - t0
        print(f"Time: {elapsed/60:.1f} min elapsed")

    # Summary
    total = time.time() - t0
    successful = [r for r in all_results if r.get('status') == 'success']
    print(f"\n{'='*60}")
    print(f"COMPLETED: {len(successful)}/{len(all_results)} experiments in {total/60:.1f} min")
    print(f"{'='*60}")

    for r in successful:
        print(f"  {r['model_type']:15s} h={r['horizon']} s={r['seed']}  "
              f"Test F1={r['test_macro_f1']:.4f}  Acc={r['test_accuracy']:.4f}")


if __name__ == '__main__':
    main()
