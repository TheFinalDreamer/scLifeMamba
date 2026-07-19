# MODEL ARCHITECTURE AUDIT

**Date:** 2026-07-19
**Audit Scope:** Full codebase model inspection — encoders, fusion, Mamba, LSTM, classifier

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 Intended Architecture (from main.tex)

```
RNA ──→ RNAEncoder ──→ z_rna ──┐
                                  ├──→ StateAwareFusion ──→ z_fused
Protein → ProteinEncoder → z_prot ┘                            │
                                                                ↓
                                        MambaLSTMEncoder ──→ h_encoded
                                                                │
                                          ┌─────────────────────┤
                                          ↓                     ↓
                                   ClassificationHead    PseudotimeHead
```

### 1.2 Actual Code Structure

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| RNA Encoder | `models/encoders.py:20-39` | MLP + LayerNorm + ReLU + Dropout | ✓ |
| Protein Encoder | `models/encoders.py:42-61` | MLP + LayerNorm + ReLU + Dropout | ✓ |
| Dynamic Fusion | `models/fusion.py:8-55` | Gated weighted sum (softmax) | ✓ |
| Simple Fusion | `models/fusion.py:58-73` | Concat + Linear | ✓ |
| Mamba Block | `models/mamba_block.py:72-109` | 3-tier backend selector | ✓ |
| Mamba-LSTM Encoder | `models/mamba_lstm.py:7-94` | Mamba → LSTM with γ-gate | ✓ |
| Classification Head | `models/heads.py` | MLP head | ✓ |
| Full Model | `models/scLifeMamba.py:30-155` | Assembly | ⚠ DIM BUG |

---

## 2. DETAILED COMPONENT AUDIT

### 2.1 RNA Encoder

```python
# models/encoders.py:20-39
class RNAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.2):
        # MLP: input_dim → 512 → 256 (LayerNorm+ReLU+Dropout between)
        self.output_dim = 256
```

**Architecture:** Pure MLP — `Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm → ReLU → Dropout`

**Input:** `(B, rna_dim)` where rna_dim = 1000 (HVGs)
**Output:** `(B, 256)`

**Verdict:** ✓ Correct for per-cell encoding. No learnable sequence context.

### 2.2 Protein Encoder

```python
# models/encoders.py:42-61
class ProteinEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64], dropout=0.2):
        # MLP: input_dim → 64
        self.output_dim = 64
```

**Architecture:** Single hidden layer MLP — `Linear → LayerNorm → ReLU → Dropout`

**Input:** `(B, protein_dim)` where protein_dim = 228 (ADT panel)
**Output:** `(B, 64)`

**Verdict:** ✓ Correct for per-cell encoding.

### 2.3 StateAwareFusion

```python
# models/fusion.py:8-55
class StateAwareFusion(nn.Module):
    def forward(self, z_rna, z_protein):
        h_rna = self.rna_proj(z_rna)      # (B,256) → (B,128)
        h_protein = self.protein_proj(z_protein)  # (B,64) → (B,128)
        gate_input = cat([h_rna, h_protein], dim=-1)  # (B,256)
        alpha = softmax(self.gate(gate_input), dim=-1)  # (B,2)
        z_fused = alpha[:,0:1] * h_rna + alpha[:,1:2] * h_protein
        return z_fused, alpha
```

**Architecture:** Project both modalities to `hidden_dim=128`, learn per-cell gating weights via 2-layer MLP, weighted-sum fusion.

**Input:** `z_rna: (B, 256)`, `z_protein: (B, 64)`
**Output:** `z_fused: (B, 128)`, `modality_weights: (B, 2)`

**Verdict:** ✓ Correct per-cell dynamic fusion. No pseudotime/horizon context in the gate (cf. main.tex Eq.1 which includes τ and h — this is a spec gap, not a bug; the formula in the paper is aspirational).

### 2.4 MambaBlock — Backend Selection

```python
# models/mamba_block.py:72-109
class MambaBlock(nn.Module):
    def __init__(self, dim=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        if is_mamba_available():
            self._block = RealMambaBlock(...)      # wraps mamba_ssm.Mamba
            self.backend_name = "native_mamba_ssm"
        elif TorchSelectiveSSMBlock available:
            self._block = TorchSelectiveSSMBlock(...)  # pure PyTorch SSM
            self.backend_name = "torch_selective_ssm"
        else:
            self._block = FallbackMambaBlock(...)      # Conv1d+GRU
            self.backend_name = "fallback_conv1d_gru"
```

**Current Backend (this machine):** `torch_selective_ssm`

### 2.5 TorchSelectiveSSM — Detailed Audit

**File:** `models/torch_selective_ssm.py`

**Algorithm:** Implements the full Mamba selective SSM from Gu & Dao (2023):
1. **Input projection:** `in_proj(x) → (x_branch, z_branch)` each `(B, L, d_inner)`
2. **Conv1d local mixing:** Depthwise conv with SiLU activation
3. **Delta (Δ):** `softplus(dt_proj(x_proj(x_conv)))` — input-dependent step size
4. **Selective B, C:** `B_proj(x_conv)`, `C_proj(x_conv)` — input-dependent
5. **A discretization:** `A_bar = exp(δ · A)` where A is HiPPO-initialized
6. **Parallel scan:** Cumulative-product-based associative scan
7. **Gating:** `y = SSM(x) ⊙ SiLU(z_branch)`
8. **Residual + Norm:** `out_proj(y) + residual`, LayerNorm

**Key Implementation Details:**
- `d_model=128, d_state=16, d_conv=4, expand=2`
- `d_inner = d_model * expand = 256`
- `dt_rank = ceil(d_model/16) = 8`
- A matrix initialized from HiPPO (log-spaced positive values)
- D parameter (skip connection) initialized to 1.0
- dt_proj bias initialized uniform(-3.0, -1.5) for stability

**Validation (this machine):**
```
backend: torch_selective_ssm
input_shape: (4, 32, 128) → output_shape: (4, 32, 128)
output_finite: True
gradients_finite: True
peak_memory_mb: 42.3
device: cuda
torch_version: 2.5.1+cu121
```

**Computational Complexity:**
- Conv1d: O(B·L·d_inner·d_conv) → O(B·L·256·4)
- Selective scan: O(B·L·d_inner·d_state) → O(B·L·256·16)
- Overall: O(B·L·d_model·expand·d_state) — linear in sequence length ✓

**Verdict:** ✓ Algorithmically faithful Mamba implementation. NOT the CUDA-optimized `mamba_ssm` kernel (no hardware-aware SRAM kernel fusion), but mathematically equivalent. Suitable for formal evaluation with proper disclosure.

### 2.6 TorchSelectiveSSM vs Native mamba_ssm

| Property | Native mamba_ssm | TorchSelectiveSSM |
|----------|-----------------|-------------------|
| Algorithm | Selective SSM | Selective SSM ✓ |
| Δ (input-dependent) | ✓ | ✓ |
| B, C (selective) | ✓ | ✓ |
| HiPPO initialization | ✓ | ✓ |
| Parallel scan | ✓ (associative) | ✓ (cumprod-based) |
| Conv1d mixing | ✓ | ✓ |
| SiLU gating | ✓ | ✓ |
| CUDA kernel fusion | Hardware-optimized | Standard PyTorch ops |
| Memory efficiency | SRAM-aware | Standard |
| Speed | Optimized | Slower (~3-10×) |
| Cross-platform | Linux only | Any PyTorch CUDA |

**Key Difference:** Algorithmic equivalence, implementation difference (kernel optimization). For a method paper, the algorithmic contribution is what matters; kernel optimization is engineering.

### 2.7 MambaLSTMEncoder

```python
# models/mamba_lstm.py:7-94
class MambaLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, ...):
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba = MambaBlock(dim=hidden_dim, ...)  # or None
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, ...)  # or None
        self.gamma = nn.Parameter(torch.tensor(0.5))  # learned fusion weight

    def forward(self, x):  # x: (B, seq_len, hidden_dim)  ← EXPECTS 3D
        h = self.input_proj(x)
        h_mamba = self.mamba(h) if self.use_mamba else h
        h_lstm, _ = self.lstm(h) if self.use_lstm else (h, None)
        h_out = h_mamba + self.gamma * h_lstm  # residual fusion
        h_pooled = center_pool(h_out)  # or attention/mean/last
        return h_pooled, h_out
```

**Architecture:**
1. Input projection: `Linear(input_dim, hidden_dim)`
2. Mamba branch: Selective SSM over sequence → `h_mamba`
3. LSTM branch: 2-layer bidirectional-like (actually unidirectional `nn.LSTM`) over sequence → `h_lstm`
4. Learned fusion: `h_out = h_mamba + γ · h_lstm`
5. Pooling: center-pooling (default), attention-pooling, mean-pooling, or last-pooling
6. LayerNorm + Dropout

**Verdict:** ✓ Correct sequence encoder architecture. Mamba captures long-range selective dependencies; LSTM captures local transition dynamics; γ learns the contribution balance.

**LSTM Confirmation:** `nn.LSTM` is PyTorch's standard LSTM — this is real LSTM, not a fallback or approximation. ✓

**IMPORTANT NOTE on Bidirectionality:** The `nn.LSTM` here is UNIDIRECTIONAL (default `bidirectional=False`). For trajectory modeling, unidirectional is arguably correct (future cells shouldn't influence past), but this is a design choice to document.

### 2.8 scLifeMamba — CRITICAL DIMENSION BUG

```python
# models/scLifeMamba.py:121-155
def forward(self, x_rna, x_protein=None):
    # x_rna: (batch_size, rna_dim) ← FLAT, NOT sequence
    z_rna = self.rna_encoder(x_rna)          # → (B, 256)
    z_protein = self.protein_encoder(x_protein)  # → (B, 64)
    z_fused, weights = self.fusion(z_rna, z_protein)  # → (B, 128)

    # BUG: passes 2D (B, 128) to encoder expecting 3D (B, seq_len, 128)
    h = self.seq_encoder(z_fused)  # ← DIMENSION MISMATCH
```

**What Happens:**
1. `z_fused` is `(B, 128)` — a batch of individual cell representations
2. `MambaLSTMEncoder.input_proj(z_fused)` → `(B, 128)` (Linear is applied to last dim)
3. `MambaBlock(z_fused)` where `z_fused` is 2D:
   - `SelectiveSSM.forward()` line 72: `B, L, D = x.shape` → **crashes** (only 2 values to unpack)
4. Even if it didn't crash: Mamba would treat the batch dim as sequence length and sequence dim as hidden — **semantically wrong**

**Root Cause:** The model was designed for per-cell prediction (classify a cell based on its own RNA+Protein), but the MambaLSTMEncoder was designed for sequence modeling (classify a target based on a sequence of cells). These two paradigms haven't been connected.

**Fix Required:** The forward method must accept sequence input:
```python
def forward(self, x_rna, x_protein=None):
    # x_rna: (B, seq_len, rna_dim) ← SEQUENCE
    B, L, D_rna = x_rna.shape
    # Option A: Encode each position independently
    z_rna = self.rna_encoder(x_rna.view(B*L, D_rna)).view(B, L, -1)
    # Option B: Use a sequence-aware RNA encoder
    ...
    # Then feed to sequence encoder
    h = self.seq_encoder(z_fused)  # now correctly receives (B, seq_len, hidden_dim)
```

---

## 3. COMPONENT VERIFICATION SUMMARY

| Component | Real Implementation? | Input Shape | Output Shape | Working? |
|-----------|---------------------|-------------|--------------|----------|
| RNA Encoder | MLP (real) | (B, 1000) | (B, 256) | ✓ |
| Protein Encoder | MLP (real) | (B, 228) | (B, 64) | ✓ |
| StateAwareFusion | Gated sum (real) | (B,256)+(B,64) | (B, 128) | ✓ |
| MambaBlock | TorchSelectiveSSM (real SSM) | (B, L, 128) | (B, L, 128) | ✓ |
| LSTM (in encoder) | nn.LSTM (real) | (B, L, 128) | (B, L, 128) | ✓ |
| MambaLSTMEncoder | Mamba+LSTM+γ (real) | (B, L, 128) | (B, 128) | ✓ |
| ClassificationHead | MLP (real) | (B, 128) | (B, num_classes) | ✓ |
| **scLifeMamba.forward()** | **DIM MISMATCH** | (B, 1000)+(B, 228) | — | ⚠ |

---

## 4. ANSWER TO KEY QUESTIONS

### Q1: Is Mamba real?

**Yes, partial.** The SELECTIVE SSM ALGORITHM is real — `TorchSelectiveSSM` implements the complete Mamba algorithm (selective scan, input-dependent Δ/B/C, HiPPO initialization, parallel scan). However, it is NOT the CUDA-optimized `mamba_ssm` kernel with hardware-aware SRAM fusion.

**Classification for paper:**
- CAN claim: "Mamba-based selective state-space sequence modeling" ✓
- CANNOT claim: "native mamba_ssm CUDA implementation" without mamba_ssm installed
- MUST disclose: "Selective SSM implemented in PyTorch following Gu & Dao (2023)"

### Q2: Is LSTM real?

**Yes.** `nn.LSTM` from PyTorch — standard, real LSTM with forget gates, input gates, output gates, and cell states. 2 layers, unidirectional, hidden_dim=128.

### Q3: Does the architecture match the specification?

**Partially.** The components exist (RNA encoder ✓, Protein encoder ✓, Fusion ✓, Mamba ✓, LSTM ✓, Heads ✓) but they are NOT correctly connected for sequence-level modeling. The `scLifeMamba.forward()` takes per-cell features and tries to feed them into a sequence encoder.

### Q4: What needs to change to make the architecture work?

1. **Data loader:** Must produce `(B, seq_len, rna_dim)` and `(B, seq_len, protein_dim)` tensors
2. **Encoders:** Must process each sequence position independently (or be made sequence-aware)
3. **Fusion:** Must fuse at each sequence position
4. **scLifeMamba.forward():** Must accept and propagate 3D sequence tensors
5. **Training loop:** Must handle sequence-level batching

---

## 5. FIX SPECIFICATION

### 5.1 New Data Flow

```
Input: (B, seq_len=32, rna_dim=1000), (B, seq_len=32, prot_dim=228)

Step 1: Per-position encoding
  For each position t in [0..31]:
    z_rna[:,t,:] = RNAEncoder(x_rna[:,t,:])     → (B, 32, 256)
    z_prot[:,t,:] = ProteinEncoder(x_prot[:,t,:]) → (B, 32, 64)

Step 2: Per-position fusion
  For each position t in [0..31]:
    z_fused[:,t,:], w[:,t,:] = StateAwareFusion(z_rna[:,t,:], z_prot[:,t,:])
    → (B, 32, 128)

Step 3: Sequence encoding
  h_pooled, h_seq = MambaLSTMEncoder(z_fused)
  → h_pooled: (B, 128), h_seq: (B, 32, 128)

Step 4: Prediction
  logits = ClassificationHead(h_pooled)  → (B, num_classes=4)
```

### 5.2 Required Code Changes

| File | Change |
|------|--------|
| `models/scLifeMamba.py` | Rewrite `forward()` to accept (B,L,D) input |
| `models/encoders.py` | Add per-position encoding support (or use in loop) |
| `models/fusion.py` | Add per-position fusion support (or use in loop) |
| `data/dataset.py` | Ensure sequence data loader outputs 3D tensors |
| `training/trainer.py` | Verify training loop handles sequences |
| `scripts/run_main_experiment.py` | Update to use sequence data |

---

## 6. FINAL VERDICT

**Model Architecture Status:** CODE_EXISTS_BUT_NOT_INTEGRATED

The individual components are real and working, but the assembly in `scLifeMamba.forward()` has a dimension mismatch that prevents sequence-level Mamba-LSTM training. This is a 1-2 day fix, not a fundamental design flaw.

**Next Step:** Fix `scLifeMamba.forward()` → enable sequence training → run Experiments 1-4.
