# MAMBA METHOD RESTART AUDIT

**Date:** 2026-07-19
**Audit Type:** Pre-refactoring comprehensive state assessment
**Target:** Transition from "leakage-controlled multimodal evaluation framework" → "trajectory-aware Mamba-LSTM method paper"

---

## 1. CURRENT PAPER STATE

### 1.1 Manuscript Identity

| Field | Value |
|-------|-------|
| **File** | `manuscript/bioinformatics_submission_draft_v1/main.tex` |
| **Current Title** | scLifeMamba: trajectory-aware multimodal sequence modeling for donor-held-out single-cell state analysis |
| **Target Title** | Trajectory-Aware Mamba-LSTM Framework for Multimodal Single-Cell State Modeling |
| **Current Framing** | Evaluation framework paper — emphasizes protocol, leakage control, baselines |
| **Target Framing** | Method paper — emphasizes Mamba-LSTM architecture, multimodal fusion, trajectory modeling |
| **Current Claims** | "framework provides a reproducible evaluation strategy rather than completed native Mamba performance evidence" |

### 1.2 Current Abstract (Key Sentences)

```
"We present scLifeMamba, a trajectory-aware multimodal evaluation framework..."
"The framework provides a reproducible evaluation strategy rather than completed native Mamba performance evidence."
```

**Problem:** The abstract explicitly says the paper is NOT a Mamba-LSTM method paper. For target framing, this must be rewritten to emphasize the Mamba-LSTM architecture.

### 1.3 Current Results

The paper currently reports ONLY baseline results:

| Model | Macro F1 | Accuracy |
|-------|----------|----------|
| LR RNA-only | 0.8823 ± 0.0003 | 0.9063 ± 0.0003 |
| LR Protein-only | 0.7593 ± 0.0002 | 0.7912 ± 0.0002 |
| LR RNA+Protein | 0.8796 ± 0.0001 | 0.9044 ± 0.0001 |
| RF RNA+Protein | 0.8140 ± 0.0041 | 0.8707 ± 0.0020 |
| MLP RNA+Protein | 0.8890 ± 0.0051 | 0.9121 ± 0.0039 |

**NO Mamba-LSTM, Mamba-only, LSTM-only, or Transformer results exist.**

### 1.4 Current Figures

The manuscript references 3 figures (fig1.pdf, fig2.pdf, fig3.pdf) that exist in the archived versions. Current v1 draft may not have all three.

---

## 2. EXISTING ASSETS (REAL, VERIFIED)

### 2.1 Leakage-Safe Protocol ✓

- Donor-level split before ALL preprocessing
- Train-only HVG selection, scaling, PCA, k-NN, DPT
- Held-out pseudotime mapped from training cells
- Target cell excluded from context window
- No cross-split sequence construction
- 643,384 target-isolated sequences across 4 horizons
- Diagnostics confirmed: all PASS

### 2.2 Baseline Results ✓

- 3 seeds (42, 123, 456) × 4 horizons (1, 4, 8, 16) = 12 runs per model
- Models: LR (RNA-only, Protein-only, RNA+Protein), RF (RNA+Protein), MLP (RNA+Protein)
- Full per-class metrics, confusion matrices saved
- Verified location: `outputs/leakage_safe_rerun/baselines_v2/`

### 2.3 Direction Label Audit ✓

- Naive pseudotime-delta: 99.87% of windows in one class → TRIVIAL TASK
- Lifecycle-transition scheme: max class proportion 37.1% → BALANCED
- Outputs saved: `outputs/revised_direction_labels/`

### 2.4 True-vs-Shuffled Label Diagnostic ✓

- Verified: models above random baseline (Macro F1 ~0.24)
- True labels give Macro F1 ~0.88; shuffled labels drop to ~0.24
- Confirms task is learnable
- Outputs: `outputs/leakage_safe_rerun/diagnostics/true_shuffled_label/`

---

## 3. MISSING ASSETS (CRITICAL GAPS)

### 3.1 Native Mamba ✗

| Requirement | Status |
|-------------|--------|
| `mamba_ssm` package | NOT INSTALLED (Linux-only, Windows blocked by tilelang dependency) |
| `causal-conv1d` package | NOT INSTALLED (same Linux-only constraint) |
| TorchSelectiveSSM (pure PyTorch impl) | AVAILABLE & VALIDATED on CUDA |
| MambaBlock backend | `torch_selective_ssm` (algorithmically correct, NOT native CUDA kernel) |

**Rating:** TorchSelectiveSSM is a legitimate selective SSM implementation, but it is NOT `mamba_ssm`. The paper must be explicit about which backend is used.

### 3.2 Mamba-LSTM Training Results ✗

- No training of `scLifeMamba` model under leakage-safe protocol
- No Mamba-only ablation
- No LSTM-only ablation
- No Transformer comparison
- No sequence model comparison of any kind

### 3.3 Architecture Gap: Dimension Mismatch ✗

**CRITICAL FINDING:** `scLifeMamba.forward()` in `code/src/models/scLifeMamba.py:121-155` takes per-cell features `x_rna: (B, rna_dim)` and `x_protein: (B, protein_dim)`, encodes them to `z_fused: (B, hidden_dim)`, then passes this 2D tensor to `MambaLSTMEncoder` which expects `(B, seq_len, hidden_dim)`.

The `MambaLSTMEncoder.input_proj` is `nn.Linear(input_dim, hidden_dim)` which will operate on the last dimension, so `(B, hidden_dim) → (B, hidden_dim)`. But then:
- `self.mamba(x)` where x is 2D → Mamba/SSM expects 3D `(B, L, D)` → **will fail at Conv1d or selective scan**
- `self.lstm(x)` where x is 2D → LSTM expects 3D `(B, L, D)` → **will fail**

**This means the current `scLifeMamba` model CANNOT be used as-is for sequence modeling.** The per-cell encoder path does not feed sequences into the sequence encoder.

### 3.4 Experiments Never Run

| Experiment | Status | What's Missing |
|-----------|--------|----------------|
| Exp 1: Backbone comparison | NOT RUN | MLP vs Transformer vs LSTM vs Mamba vs Mamba-LSTM |
| Exp 2: Architecture ablation | NOT RUN | Full vs -Mamba vs -LSTM vs -RNA vs -Protein |
| Exp 3: Sequence dependency | NOT RUN | Original vs shuffled vs no-order |
| Exp 4: Generalization (deep) | NOT RUN | Mamba-LSTM donor-held-out; LODO |

---

## 4. CODE ARCHITECTURE INVENTORY

### 4.1 Model Files (Real, Working)

| File | Purpose | Status |
|------|---------|--------|
| `code/src/models/encoders.py` | RNA/Protein MLP encoders | ✓ Working |
| `code/src/models/fusion.py` | StateAwareFusion + SimpleFusion | ✓ Working |
| `code/src/models/mamba_block.py` | MambaBlock (3-tier backend) | ✓ Working |
| `code/src/models/torch_selective_ssm.py` | PyTorch selective SSM | ✓ Validated on CUDA |
| `code/src/models/mamba_lstm.py` | MambaLSTMEncoder | ✓ Working (expects 3D input) |
| `code/src/models/heads.py` | Classification/Pseudotime/Embedding heads | ✓ Working |
| `code/src/models/scLifeMamba.py` | Full model assembly | ⚠ Dimension mismatch (see §3.3) |
| `code/src/models/baselines.py` | LR, RF, MLP baselines | ✓ Working |

### 4.2 Data Pipeline

| File | Purpose | Status |
|------|---------|--------|
| `code/src/data/preprocessing.py` | Leakage-safe preprocessing | ✓ Working |
| `code/src/data/split.py` | Donor-level split | ✓ Working |
| `code/src/data/dataset.py` | Sequence dataset construction | ✓ Working |
| `code/src/data/trajectory_sequence_dataset.py` | Trajectory sequence builder | ✓ Working |

### 4.3 Training Infrastructure

| File | Purpose | Status |
|------|---------|--------|
| `code/src/training/trainer.py` | Training loop | ✓ Working |
| `code/src/training/early_stopping.py` | Early stopping | ✓ Working |
| `code/src/training/scheduler.py` | LR scheduler | ✓ Working |
| `code/src/evaluation/metrics.py` | Metrics computation | ✓ Working |

---

## 5. ENVIRONMENT STATUS

| Component | Version/Status |
|-----------|---------------|
| **OS** | Windows 11 Pro 10.0.22621 |
| **GPU** | RTX 4080 16GB |
| **Python** | 3.10.6 |
| **PyTorch** | 2.5.1+cu121 |
| **CUDA Toolkit** | 12.9 (nvcc available) |
| **CUDA (torch)** | 12.1 |
| **mamba_ssm** | ✗ NOT INSTALLED (Linux-only) |
| **causal-conv1d** | ✗ NOT INSTALLED (Linux-only) |
| **TorchSelectiveSSM** | ✓ Validated (CUDA, 42.3MB peak) |
| **WSL2** | Not checked |

### 5.1 Mamba Backend Verdict

**Native `mamba_ssm` cannot be installed on Windows.** The dependency chain (`mamba-ssm → tilelang → scikit-build-core`) fails. This is a known limitation.

**Available options:**
1. **TorchSelectiveSSM** (current) — Pure PyTorch selective SSM, algorithmically faithful to Gu & Dao 2023. Valid for formal evaluation with proper disclosure.
2. **WSL2 Ubuntu** — Could potentially install native `mamba_ssm`. Not yet attempted.
3. **Remote Linux server** — External server, not currently configured.

**Recommendation for paper:** Use TorchSelectiveSSM and explicitly state: "Selective SSM implemented in pure PyTorch following the Mamba algorithm (Gu & Dao, 2023), validated on CUDA. Native mamba_ssm CUDA kernels not available on the development platform (Windows)."

---

## 6. WHAT MUST BE FIXED BEFORE "METHOD PAPER" CLAIM

### 6.1 P0: Fix Dimension Mismatch in scLifeMamba

The model must be refactored so that:
- Input: `(B, seq_len, rna_dim)` and `(B, seq_len, protein_dim)` as sequences
- RNA/Protein encoders process each position independently (or with sequence context)
- Fusion happens per-position
- MambaLSTMEncoder receives proper 3D tensors

### 6.2 P0: Run Experiment 1 (Backbone Comparison)

Must train and evaluate:
- MLP (sequence-level pooling + MLP)
- Transformer (positional encoding + self-attention)
- LSTM (bidirectional)
- Mamba (TorchSelectiveSSM)
- Mamba-LSTM (TorchSelectiveSSM + LSTM)

All under identical: split, seed set, horizons, preprocessing, optimizer, scheduler.

### 6.3 P0: Run Experiment 2 (Architecture Ablation)

Must compare:
- Full Mamba-LSTM
- Mamba-LSTM without Mamba (LSTM-only)
- Mamba-LSTM without LSTM (Mamba-only)
- Mamba-LSTM without RNA (Protein-only)
- Mamba-LSTM without Protein (RNA-only)

### 6.4 P0: Run Experiment 3 (Sequence Dependency)

Must compare:
- Original pseudotime-ordered sequence
- Randomly shuffled sequence
- No temporal order (bag-of-cells)

### 6.5 P0: Run Experiment 4 (Generalization with Deep Models)

- Mamba-LSTM under donor-held-out
- Report per-horizon breakdown

### 6.6 P1: Statistical Analysis

- Mean ± SD over 3+ seeds
- Paired statistical test between models
- NOT single-best-run reporting

---

## 7. EXPERIMENT EXECUTION PLAN

### 7.1 Output Structure

```
outputs/mamba_final/
├── exp1_backbone/
│   ├── mlp/seed42/h1/ {config.yaml, metrics.json, predictions.csv, training_log.txt}
│   ├── transformer/...
│   ├── lstm/...
│   ├── mamba/...
│   └── mamba_lstm/...
├── exp2_ablation/
│   ├── full/...
│   ├── no_mamba/...
│   ├── no_lstm/...
│   ├── no_rna/...
│   └── no_protein/...
├── exp3_sequence/
│   ├── original/...
│   ├── shuffled/...
│   └── no_order/...
└── exp4_generalization/
    └── donor_held_out/...
```

### 7.2 Estimated Runtime (RTX 4080 16GB)

| Experiment | Models | Seeds × Horizons | Est. Time |
|-----------|--------|-------------------|-----------|
| Exp 1: Backbone | 5 | 3 × 4 = 12 each | ~2-4 hours |
| Exp 2: Ablation | 6 | 3 × 4 = 12 each | ~3-5 hours |
| Exp 3: Sequence | 3 | 3 × 4 = 12 each | ~1-2 hours |
| Exp 4: Generalization | 1 | 3 × 4 = 12 | ~1 hour |
| **Total** | | | **~7-12 hours** |

---

## 8. MANUSCRIPT REFACTORING PLAN

### 8.1 Sections to Rewrite

| Section | Current | Target |
|---------|---------|--------|
| **Title** | "trajectory-aware multimodal sequence modeling for donor-held-out single-cell state analysis" | "Trajectory-Aware Mamba-LSTM Framework for Multimodal Single-Cell State Modeling" |
| **Abstract** | Evaluation framework emphasis | Mamba-LSTM architecture + multimodal + results |
| **Introduction** | Protocol-focused | Why long-range + local + multimodal needed |
| **Methods** | Lagrangian, evaluation design | Mamba SSM, LSTM refinement, trajectory-aware encoding |
| **Results 3.1** | "Leakage-controlled evaluation framework" | Keep as protocol section |
| **Results 3.2** | "Modality contribution" | → "Mamba-LSTM architecture comparison" (if results exist) |
| **Results 3.3** | "Baseline comparison" | → "Multimodal contribution analysis" |
| **Results 3.4** | "Ablation and limitations" | → "Ablation studies" (with actual ablation results) |
| **Results 3.5** | N/A | → "Generalization analysis" |
| **Discussion** | Guarded, self-limiting | Include limitations but with Mamba-LSTM findings |

### 8.2 Language Rules

| Current Phrase | Replacement |
|---------------|-------------|
| "evaluation framework" | "method" or "architecture" |
| "framework provides...rather than" | REMOVE — state what WAS done |
| "the current evidence does not establish" | State actual results with uncertainty |
| "Lagrangian" | "trajectory-aware" |
| "future native-Mamba evaluation" | "Mamba-based selective state-space modeling" |

---

## 9. RISK ASSESSMENT

| Risk | Severity | Mitigation |
|------|----------|------------|
| TorchSelectiveSSM not accepted as "real Mamba" | MEDIUM | Explicit disclosure; cite algorithm faithfulness |
| Dimension mismatch blocks all training | HIGH | Must fix before any experiment |
| Experiments produce negative results (Mamba ≤ MLP) | MEDIUM | Report honestly; discussion becomes about when Mamba helps |
| Training time exceeds estimate | LOW | RTX 4080 16GB is adequate |
| No mamba_ssm on Windows | ACCEPTED | Use TorchSelectiveSSM with disclosure |
| Single dataset limitation | ACCEPTED | Disclose in limitations |

---

## 10. BLOCKING ITEMS SUMMARY

| # | Item | Blocks |
|---|------|--------|
| 1 | Fix `scLifeMamba.forward()` dimension mismatch | All Mamba-LSTM experiments |
| 2 | Build sequence-aware data loader (B, seq_len, dim) input | All sequence model experiments |
| 3 | Write experiment script for Exp 1-4 | All results |
| 4 | Run Exp 1 (Backbone comparison) | Table 1, Figure 3 |
| 5 | Run Exp 2 (Architecture ablation) | Table 2, Figure 4 |
| 6 | Run Exp 3 (Sequence dependency) | Table 3 |
| 7 | Run Exp 4 (Generalization) | Supplementary Table |
| 8 | Run statistical tests | All tables |
| 9 | Generate Figures 1-4 | Manuscript |
| 10 | Rewrite manuscript sections | Final submission |

---

## 11. AUDIT CONCLUSION

**Current Status:** PAPER_IS_EVALUATION_FRAMEWORK_NOT_METHOD

**What Exists (Real):**
- Rigorous leakage-safe protocol ✓
- Baseline results (LR, RF, MLP) ✓
- Working TorchSelectiveSSM on CUDA ✓
- Complete data pipeline ✓
- Model architecture code (with dimension bug) ⚠

**What Does NOT Exist (Must Build):**
- Working sequence-level scLifeMamba training
- Any Mamba-LSTM experimental results
- Backbone comparison (Exp 1)
- Architecture ablation (Exp 2)
- Sequence dependency validation (Exp 3)
- Deep model generalization (Exp 4)

**Path Forward:**
1. Fix dimension mismatch → enable sequence model training
2. Run Experiments 1-4 → produce all tables/figures
3. Rewrite manuscript → method paper framing
4. If experiments show Mamba-LSTM advantage over baselines → claim "outperformed"
5. If experiments show parity → claim "achieved competitive performance with architectural advantages in long-range modeling"
6. If experiments fail → BLOCKED_WITH_REASON, report honestly
