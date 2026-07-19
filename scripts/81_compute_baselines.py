#!/usr/bin/env python3
"""Full leakage-safe baseline matrix: 5 models x 4 horizons x 3 seeds."""
import os, sys, json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "leakage_safe_v1"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "leakage_safe_rerun" / "baselines_v2"
SEEDS = [42, 123, 456]
HORIZONS = [1, 4, 8, 16]

MODELS = {
    "LR_RNA_only": lambda: ("LR", LogisticRegression(max_iter=500, n_jobs=-1), "rna"),
    "LR_Protein_only": lambda: ("LR", LogisticRegression(max_iter=500, n_jobs=-1), "protein"),
    "LR_RNA_Protein": lambda: ("LR", LogisticRegression(max_iter=500, n_jobs=-1), "both"),
    "RF_RNA_Protein": lambda: ("RF", RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=None), "both"),
    "MLP_RNA_Protein": lambda: ("MLP", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True, validation_fraction=0.1, random_state=None), "both"),
}

def train_eval(name, clf, X_train, y_train, X_test, y_test, modality, rna_dim):
    if modality == "rna":
        Xt, Xe = X_train[:, :rna_dim], X_test[:, :rna_dim]
    elif modality == "protein":
        Xt, Xe = X_train[:, rna_dim:], X_test[:, rna_dim:]
    else:
        Xt, Xe = X_train, X_test
    clf.fit(Xt, y_train)
    y_pred = clf.predict(Xe)
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'per_class_f1': [float(x) for x in f1_score(y_test, y_pred, average=None)],
        'confusion_matrix': [[int(x) for x in row] for row in confusion_matrix(y_test, y_pred)],
        'predictions': y_pred.tolist(),
        'y_true': y_test.tolist(),
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run leakage-safe classical baselines.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    processed_dir = args.data_dir
    out_dir = args.output_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    features = np.load(processed_dir / 'features_combined.npy')
    seq_df = pd.read_csv(processed_dir / 'sequence_manifest.csv')
    rna_dim = 1000
    all_runs = []

    total = len(SEEDS) * len(HORIZONS) * len(MODELS)
    n_done = 0

    for seed in SEEDS:
        for h in HORIZONS:
            h_mask = seq_df['horizon'] == h
            h_seq = seq_df[h_mask]
            target_indices = h_seq['target_idx'].values.astype(int)
            X = features[target_indices]
            y = h_seq['target_label'].values

            train_mask = (h_seq['split'] == 'train').values
            test_mask = (h_seq['split'] == 'test').values
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            for model_name, factory in MODELS.items():
                n_done += 1
                model_type, clf_factory, modality = factory()
                if model_type in ("RF", "MLP"):
                    clf = clf_factory
                    if hasattr(clf, 'random_state'):
                        clf.random_state = seed
                else:
                    clf = clf_factory

                t0 = time.time()
                try:
                    result = train_eval(model_name, clf, X_train_s, y_train, X_test_s, y_test, modality, rna_dim)
                    result.update({
                        'model': model_name, 'seed': seed, 'horizon': h,
                        'runtime_s': round(time.time() - t0, 1),
                        'status': 'success', 'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
                    })
                except Exception as e:
                    result = {'model': model_name, 'seed': seed, 'horizon': h,
                              'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}

                all_runs.append(result)
                if result['status'] == 'success':
                    print(f"[{n_done}/{total}] {model_name} seed={seed} h={h}: MacroF1={result['macro_f1']:.4f}")
                else:
                    print(f"[{n_done}/{total}] {model_name} seed={seed} h={h}: FAILED — {result['error'][:80]}")

    # Save all runs
    runs_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('confusion_matrix', 'per_class_f1', 'predictions', 'y_true')}
                            for r in all_runs])
    runs_df.to_csv(out_dir / 'all_runs.csv', index=False)

    # Save per-horizon per-class F1
    pcf1_rows = []
    for r in all_runs:
        if r['status'] != 'success': continue
        for ci, f1v in enumerate(r.get('per_class_f1', [])):
            pcf1_rows.append({'model': r['model'], 'seed': r['seed'], 'horizon': r['horizon'], 'class': ci, 'f1': f1v})
    pd.DataFrame(pcf1_rows).to_csv(out_dir / 'per_class_f1.csv', index=False)

    # Save confusion matrices
    cm_dir = out_dir / 'confusion_matrices'
    cm_dir.mkdir(exist_ok=True)
    for r in all_runs:
        if r['status'] != 'success': continue
        cm_df = pd.DataFrame(r['confusion_matrix'], columns=[f'pred_{i}' for i in range(4)], index=[f'true_{i}' for i in range(4)])
        cm_df.to_csv(cm_dir / f"{r['model']}_s{r['seed']}_h{r['horizon']}.csv")

    # Aggregate summaries
    success = [r for r in all_runs if r['status'] == 'success']
    df = pd.DataFrame(success)

    # By model+horizon
    mh = df.groupby(['model', 'horizon']).agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        n_seeds=('seed', 'nunique'),
    ).reset_index()
    mh.to_csv(out_dir / 'summary_by_model_horizon.csv', index=False)

    # Across seeds (per model)
    ms = df.groupby('model').agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        n_runs=('macro_f1', 'count'),
    ).reset_index()
    ms.to_csv(out_dir / 'summary_across_seeds.csv', index=False)

    # Across horizons (per model)
    mh2 = df.groupby('model').agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
    ).reset_index()
    mh2.to_csv(out_dir / 'summary_across_horizons.csv', index=False)

    print(f"\n{'='*60}")
    print("BASELINE MATRIX COMPLETE")
    print(f"Completed: {len(success)}/{total}")
    print(f"Failed: {total - len(success)}")
    print(f"Results: {out_dir}")
    print(f"\nModel summary (across seeds & horizons):")
    for _, row in ms.iterrows():
        print(f"  {row['model']}: MacroF1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
