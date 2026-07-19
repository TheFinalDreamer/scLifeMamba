# EXPERIMENT EVIDENCE MAP

**Date:** 2026-07-19
**Audit Type:** Complete evidence traceability for all paper claims
**Principle:** Every number in the manuscript must trace to a specific experiment script, config, and output file.

---

## PAPER CLAIM → EVIDENCE TRACING

### Claims in Abstract

| # | Paper Claim | Evidence File | Script | Seed/Split | Status |
|---|-----------|---------------|--------|------------|--------|
| A1 | "161,764 PBMC CITE-seq cells" | `data/processed/leakage_safe_v1/rna_hvg_all.npy` (161764, 1000) | `code/scripts/preprocess_dataset.py` | N/A | ✅ TRACED |
| A2 | "1,000 training-selected HVGs" | `data/processed/leakage_safe_v1/hvg_genes_train_only.txt` | Same | Train donors only | ✅ TRACED |
| A3 | "228 surface proteins" | `data/processed/leakage_safe_v1/protein_norm_all.npy` (161764, 228) | Same | N/A | ✅ TRACED |
| A4 | "643,384 target-isolated sequences" | `data/processed/leakage_safe_v1/sequence_manifest.csv` | `code/scripts/13_build_trajectory_sequences.py` | 4 horizons | ✅ TRACED |
| A5 | "Mamba-LSTM Macro F1 0.9268" | `code/outputs/mamba_final/exp1_backbone/mamba_lstm/seed42/h1/metrics.json` | `code/scripts/run_mamba_final_experiments.py` | h=1, s=42, 3 epochs | ⚠ SINGLE SEED |
| A6 | "Mamba-LSTM accuracy 0.9409" | Same | Same | Same | ⚠ SINGLE SEED |
| A7 | "LR RNA+Protein F1 0.8796 ± 0.0001" | `outputs/leakage_safe_rerun/baselines/aggregate_summary.json` | `code/scripts/81_compute_baselines.py` | 3 seeds × 4 horizons | ✅ TRACED |
| A8 | "Pure PyTorch selective SSM" | `code/src/models/torch_selective_ssm.py` — validated on CUDA | N/A | N/A | ✅ TRACED |

### Claims in Results

| # | Paper Claim | Evidence | Status |
|---|-----------|----------|--------|
| R1 | "MLP F1 0.9340" | `code/outputs/mamba_final/exp1_backbone/mlp/seed42/h1/metrics.json` | ⚠ SINGLE SEED, 30 epochs vs Mamba-LSTM 3 epochs |
| R2 | "LSTM-only F1 0.9254" | `code/outputs/mamba_final/fast_subset_results.json` (10k subset) | ⚠ SUBSET ONLY |
| R3 | "Mamba-only F1 0.9054" | Same (10k subset) | ⚠ SUBSET ONLY |
| R4 | "LR RNA-only F1 0.8823" | `outputs/leakage_safe_rerun/baselines/aggregate_summary.json` | ✅ |
| R5 | "LR Protein-only F1 0.7593" | Same | ✅ |
| R6 | "Direction label 99.87% one class" | `outputs/revised_direction_labels/` | ✅ |
| R7 | "Lifecycle transition max 37.1%" | Same | ✅ |

### Claims in Methods

| # | Paper Claim | Evidence | Status |
|---|-----------|----------|--------|
| M1 | "Mamba selective SSM block" | `code/src/models/torch_selective_ssm.py` — SelectiveSSM class | ✅ |
| M2 | "2-layer unidirectional LSTM" | `code/src/models/mamba_lstm.py` — nn.LSTM | ✅ |
| M3 | "Dynamic fusion gate" | `code/src/models/fusion.py` — StateAwareFusion | ✅ |
| M4 | "RNA encoder MLP" | `code/src/models/encoders.py` — RNAEncoder | ✅ |
| M5 | "Protein encoder MLP" | `code/src/models/encoders.py` — ProteinEncoder | ✅ |
| M6 | "Learned γ fusion" | `code/src/models/mamba_lstm.py` — gamma Parameter | ✅ |
| M7 | "Donor-held-out 5/2/1 split" | `data/processed/leakage_safe_v1/split_assignments.csv` | ✅ |

---

## EXPERIMENT COMPLETENESS MATRIX

### Experiment 1: Backbone Comparison

| Model | Seed 42 | Seed 123 | Seed 456 | Full-Data? | Same Epochs? |
|-------|---------|----------|----------|------------|--------------|
| MLP | ✅ F1=0.9340 | ❌ | ❌ | ✅ Full (161k) | ❌ 30 epochs |
| LSTM | ⚠ F1=0.9254 (subset) | ❌ | ❌ | ❌ 10k subset | ❌ 5 epochs |
| Mamba | ⚠ F1=0.9054 (subset) | ❌ | ❌ | ❌ 10k subset | ❌ 5 epochs |
| Mamba-LSTM | ✅ F1=0.9268 | ❌ | ❌ | ✅ Full (161k) | ❌ 3 epochs |

**CRITICAL:** No two models were trained with the same number of epochs on the same data. This makes direct comparison invalid.

### Experiment 2: Architecture Ablation

| Variant | Status |
|---------|--------|
| Full Mamba-LSTM | ✅ 1 seed, 3 epochs |
| -Mamba (LSTM-only) | ⚠ Subset only |
| -LSTM (Mamba-only) | ⚠ Subset only |
| -Protein (RNA-only) | ❌ |
| -RNA (Protein-only) | ❌ |

### Experiment 3: Sequence Dependency

| Condition | Status |
|-----------|--------|
| Original order | ⚠ 1 seed |
| Random shuffled | ❌ |
| No temporal order | ❌ |

### Experiment 4: Generalization

| Analysis | Status |
|----------|--------|
| Donor-held-out (primary) | ✅ Verified in protocol |
| LODO | ❌ |

---

## CRITICAL ISSUES IDENTIFIED

### BLOCKING

| # | Issue | Impact |
|---|-------|--------|
| **B1** | **MLP > Mamba-LSTM** (0.9340 > 0.9268) | Cannot claim Mamba-LSTM outperforms; must use "competitive" language |
| **B2** | **Unfair training comparison** — MLP 30 epochs, Mamba-LSTM 3 epochs | Results are not comparable; need same-epoch retraining |
| **B3** | **Only 1 seed for all deep models** | No error bars, no statistical tests possible |
| **B4** | **No sequence order ablation** | Cannot claim trajectory-aware benefit |

### WARNING (Non-blocking but must fix)

| # | Issue | Fix |
|---|-------|-----|
| W1 | LSTM/Mamba are subset results | Mark clearly in manuscript as preliminary |
| W2 | Abstract claims "LSTM 0.9254" without noting it's subset | Add footnote |
| W3 | No per-class F1 analysis reported | Add to supplementary |

---

## EVIDENCE INTEGRITY VERDICT

**Overall:** PARTIALLY_TRACED

All architecture claims are supported by working code. Classical baseline results are fully traced with multiple seeds/horizons. Deep learning results are preliminary (single seed, unequal training, subset data for some models).

**Required for submission:**
1. ✅ Fair comparison with equal epochs (experiment running)
2. ✅ Multi-seed results (experiment running)
3. ✅ Sequence order ablation (experiment running)
4. ⬜ Update manuscript tables with fair results
5. ⬜ Adjust language if MLP still leads after fair comparison
