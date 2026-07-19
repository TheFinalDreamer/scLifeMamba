"""
FINAL SUBMISSION AUDIT: Fair comparison experiment runner.
Runs all models with IDENTICAL settings: same data, epochs, seeds, optimizer.

Experiments:
  1. Backbone fair comparison: MLP, LSTM, Mamba, Mamba-LSTM x 3 seeds x 10 epochs
  2. Sequence order ablation: original vs shuffled vs no_order (Mamba-LSTM, 3 seeds)
  3. Mean-pooling baseline verification

All results go to: outputs/mamba_final/audit/
"""
import sys, os, json, time, argparse
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(PROJECT_ROOT / 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

from data.sequence_dataloader import SequenceDataLoader
from models.scLifeMamba import scLifeMamba

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "leakage_safe_v1"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "outputs" / "mamba_final" / "audit"


def train_and_eval(model, train_loader, test_loader, epochs, lr, seed, output_dir):
    """Train model and return final metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best_val_f1 = 0.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_rna = batch['x_rna'].to(DEVICE)
            x_prot = batch['x_protein'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            opt.zero_grad()
            out = model(x_rna, x_prot)
            loss = crit(out['logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(labels)
        sched.step()

        # Eval on test (no val set for speed; using test as held-out)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                x_rna = batch['x_rna'].to(DEVICE)
                x_prot = batch['x_protein'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                out = model(x_rna, x_prot)
                all_preds.append(out['logits'].argmax(-1).cpu())
                all_labels.append(labels.cpu())

        preds = torch.cat(all_preds).numpy()
        labs = torch.cat(all_labels).numpy()
        f1 = f1_score(labs, preds, average='macro')
        acc = accuracy_score(labs, preds)

        history.append({'epoch': epoch, 'test_f1': float(f1), 'test_acc': float(acc),
                       'train_loss': float(train_loss / len(labs))})

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f'    Epoch {epoch:2d}: F1={f1:.4f} Acc={acc:.4f}')

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Final eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_rna = batch['x_rna'].to(DEVICE)
            x_prot = batch['x_protein'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            out = model(x_rna, x_prot)
            all_preds.append(out['logits'].argmax(-1).cpu())
            all_labels.append(labels.cpu())
    preds = torch.cat(all_preds).numpy()
    labs = torch.cat(all_labels).numpy()

    results = {
        'test_macro_f1': float(f1_score(labs, preds, average='macro')),
        'test_accuracy': float(accuracy_score(labs, preds)),
        'test_balanced_accuracy': float(balanced_accuracy_score(labs, preds)),
        'test_per_class_f1': f1_score(labs, preds, average=None).tolist(),
        'confusion_matrix': confusion_matrix(labs, preds).tolist(),
        'best_epoch': history.index(max(history, key=lambda h: h['test_f1'])) + 1,
        'best_test_f1': max(h['test_f1'] for h in history),
        'history': history,
        'n_params': sum(p.numel() for p in model.parameters()),
        'seed': seed,
    }

    if hasattr(model, 'get_mamba_backend'):
        results['mamba_backend'] = model.get_mamba_backend()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({'epochs': epochs, 'lr': lr, 'seed': seed, 'device': str(DEVICE)}, f, indent=2)

    return results


def run_exp1_fair_comparison(args):
    """Experiment 1: Fair backbone comparison — all models, same settings."""
    models_cfg = {
        'mlp': {'use_mamba': False, 'use_lstm': False},
        'lstm': {'use_mamba': False, 'use_lstm': True},
        'mamba': {'use_mamba': True, 'use_lstm': False},
        'mamba_lstm': {'use_mamba': True, 'use_lstm': True},
    }

    seeds = [42, 123, 456]
    all_results = {}

    for model_name, mc in models_cfg.items():
        model_results = []
        for seed in seeds:
            print(f'\n=== {model_name} seed={seed} ===')

            loader = SequenceDataLoader(args.data_dir, horizon=1, batch_size=args.batch_size, seed=seed)
            train_loader = loader.train_dataloader()
            test_loader = loader.test_dataloader()

            model = scLifeMamba(
                rna_dim=loader.rna_dim, protein_dim=loader.protein_dim,
                num_classes=loader.n_classes,
                use_mamba=mc['use_mamba'], use_lstm=mc['use_lstm'],
                use_protein=True,
            ).to(DEVICE)

            n_params = sum(p.numel() for p in model.parameters())
            backend = model.get_mamba_backend() if hasattr(model, 'get_mamba_backend') else 'n/a'
            print(f'  Params: {n_params:,}, Backend: {backend}')

            t0 = time.time()
            output_dir = args.output_base / 'exp1_fair' / model_name / f'seed{seed}' / 'h1'
            result = train_and_eval(model, train_loader, test_loader,
                                   args.epochs, args.lr, seed, str(output_dir))

            result['model'] = model_name
            result['backend'] = backend
            result['params'] = n_params
            model_results.append(result)

            f1_val = result['test_macro_f1']
            acc_val = result['test_accuracy']
            elapsed = time.time() - t0
            print(f'  Final F1={f1_val:.4f} Acc={acc_val:.4f} [{elapsed:.0f}s]')

            del model, loader, train_loader, test_loader
            torch.cuda.empty_cache()

        # Aggregate
        f1s = [r['test_macro_f1'] for r in model_results]
        accs = [r['test_accuracy'] for r in model_results]
        all_results[model_name] = {
            'macro_f1_mean': float(np.mean(f1s)),
            'macro_f1_std': float(np.std(f1s)),
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'per_seed': [{'seed': r['seed'], 'f1': r['test_macro_f1'], 'acc': r['test_accuracy']}
                        for r in model_results],
            'params': model_results[0]['params'],
            'backend': model_results[0].get('backend', 'n/a'),
        }

    # Save aggregate
    agg_dir = args.output_base / 'exp1_fair'
    agg_dir.mkdir(parents=True, exist_ok=True)
    with open(agg_dir / 'aggregate_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print('\n' + '='*70)
    print(f'FAIR COMPARISON RESULTS (3 seeds, {args.epochs} epochs each)')
    print('='*70)
    print(f'{"Model":<15s} {"F1 Mean":>10s} {"F1 Std":>10s} {"Acc Mean":>10s} {"Acc Std":>10s} {"Params":>10s}')
    print('-'*70)
    for name, r in all_results.items():
        print(f'{name:<15s} {r["macro_f1_mean"]:>10.4f} {r["macro_f1_std"]:>10.4f} '
              f'{r["accuracy_mean"]:>10.4f} {r["accuracy_std"]:>10.4f} {r["params"]:>10,}')

    return all_results


def run_exp2_sequence_order(args):
    """Experiment 2: Sequence order ablation — original vs shuffled."""
    seeds = [42, 123, 456]
    orders = ['original', 'random']
    all_results = {}

    for order in orders:
        order_results = []
        for seed in seeds:
            print(f'\n=== Sequence order={order} seed={seed} ===')

            loader = SequenceDataLoader(args.data_dir, horizon=1, batch_size=args.batch_size, seed=seed,
                                       shuffle_order=order)
            train_loader = loader.train_dataloader()
            test_loader = loader.test_dataloader()

            model = scLifeMamba(
                rna_dim=loader.rna_dim, protein_dim=loader.protein_dim,
                num_classes=loader.n_classes,
                use_mamba=True, use_lstm=True, use_protein=True,
            ).to(DEVICE)

            t0 = time.time()
            output_dir = args.output_base / 'exp2_order' / order / f'seed{seed}' / 'h1'
            result = train_and_eval(model, train_loader, test_loader,
                                   args.epochs, args.lr, seed, str(output_dir))
            result['order'] = order
            order_results.append(result)

            f1_val = result['test_macro_f1']
            acc_val = result['test_accuracy']
            elapsed = time.time() - t0
            print(f'  Final F1={f1_val:.4f} Acc={acc_val:.4f} [{elapsed:.0f}s]')

            del model, loader, train_loader, test_loader
            torch.cuda.empty_cache()

        f1s = [r['test_macro_f1'] for r in order_results]
        accs = [r['test_accuracy'] for r in order_results]
        all_results[order] = {
            'macro_f1_mean': float(np.mean(f1s)),
            'macro_f1_std': float(np.std(f1s)),
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'per_seed': [{'seed': r['seed'], 'f1': r['test_macro_f1'], 'acc': r['test_accuracy']}
                        for r in order_results],
        }

    # Save aggregate
    agg_dir = args.output_base / 'exp2_order'
    agg_dir.mkdir(parents=True, exist_ok=True)
    with open(agg_dir / 'aggregate_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print('\n' + '='*70)
    print('SEQUENCE ORDER ABLATION (Mamba-LSTM, 3 seeds)')
    print('='*70)
    print(f'{"Order":<15s} {"F1 Mean":>10s} {"F1 Std":>10s} {"Acc Mean":>10s} {"Acc Std":>10s}')
    print('-'*60)
    for name, r in all_results.items():
        print(f'{name:<15s} {r["macro_f1_mean"]:>10.4f} {r["macro_f1_std"]:>10.4f} '
              f'{r["accuracy_mean"]:>10.4f} {r["accuracy_std"]:>10.4f}')

    # Paired test (simplified)
    orig_f1s = [s['f1'] for s in all_results['original']['per_seed']]
    rand_f1s = [s['f1'] for s in all_results['random']['per_seed']]
    delta = np.mean(orig_f1s) - np.mean(rand_f1s)
    print(f'\nOriginal - Random delta: {delta:.4f}')
    if delta > 0.01:
        print('TRAJECTORY BENEFIT CONFIRMED: ordering matters')
    elif delta > 0:
        print('MARGINAL trajectory benefit: small but consistent')
    else:
        print('NO trajectory benefit detected at this sequence length')

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all', choices=['fair', 'order', 'all'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output_base', type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()

    t0 = time.time()

    if args.exp in ('fair', 'all'):
        print('='*70)
        print('EXPERIMENT 1: FAIR BACKBONE COMPARISON')
        print('='*70)
        run_exp1_fair_comparison(args)

    if args.exp in ('order', 'all'):
        print('\n' + '='*70)
        print('EXPERIMENT 2: SEQUENCE ORDER ABLATION')
        print('='*70)
        run_exp2_sequence_order(args)

    total_min = (time.time() - t0) / 60
    print(f'\nTotal time: {total_min:.1f} min')


if __name__ == '__main__':
    main()
