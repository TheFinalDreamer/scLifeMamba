# FINAL REVIEWER SIMULATION — scLifeMamba

**Date:** 2026-07-19
**Simulated Reviewer:** Bioinformatics Adversarial Reviewer
**Review Type:** Pre-submission attack surface audit
**Verdict:** READY_FOR_SUBMISSION (with 3 mandatory fixes)

---

## REVIEWER #1 (Adversarial)

### Overall Assessment

This manuscript presents scLifeMamba, a Mamba-LSTM architecture for multimodal single-cell state modeling under donor-held-out evaluation. The paper is generally well-structured and refreshingly honest about its limitations. However, I have several concerns that must be addressed before acceptance.

---

## 1. CLAIM-EVIDENCE LINE-BY-LINE AUDIT

### 1.1 Abstract (lines 63-65)

| # | Claim Text | Evidence | Risk | Verdict |
|---|-----------|----------|------|---------|
| A1 | "integrating these modalities into trajectory-aware sequence models remains an open challenge" | Literature review, general knowledge | LOW | ✅ ACCEPT |
| A2 | "Mamba-LSTM model achieved Macro F1 of 0.9268 and accuracy of 0.9409" | `metrics.json` — full data, s=42, h=1, 3 epochs | MEDIUM | ⚠ Needs SD/seed count |
| A3 | "representing a substantial improvement over logistic regression baselines (Macro F1 0.8796 ± 0.0001)" | LR: aggregate_summary.json (12 runs); Mamba-LSTM: single run | **HIGH** | 🔴 1-seed vs 12-run baseline |
| A4 | "MLP (F1 0.9340)" | Full data, s=42, 30 epochs | MEDIUM | ⚠ 30 epochs vs Mamba-LSTM 3 epochs |
| A5 | "LSTM (F1 0.9254), Mamba (F1 0.9054)" | Subset data (10k), 5 epochs each | **HIGH** | 🔴 Mixed data scales |
| A6 | "all deep sequence models substantially exceed classical baselines" | All > 0.8796 | LOW | ✅ ACCEPT |
| A7 | "pure PyTorch and validated on CUDA, cross-platform alternative" | TorchSelectiveSSM validated | LOW | ✅ ACCEPT |

**Reviewer Comment:** The abstract mixes full-data (Mamba-LSTM: 161k seq, 3 epochs), single-seed (MLP: 161k seq, 30 epochs), and subset (LSTM/Mamba: 10k seq, 5 epochs) results. This is misleading. A reader scanning the abstract would reasonably assume these numbers come from equivalent experimental conditions. They do not. **MUST FIX: State data scale explicitly or harmonize to comparable conditions.**

### 1.2 Introduction (line 86)

| # | Claim Text | Evidence | Risk | Verdict |
|---|-----------|----------|------|---------|
| I1 | "Mamba-LSTM encoder that combines selective state-space modeling... with gated LSTM refinement" | Code verified | LOW | ✅ ACCEPT |
| I2 | "dynamic modality fusion gate that learns per-cell RNA-protein integration weights" | Code verified | LOW | ✅ ACCEPT |
| I3 | "donor-held-out protocol that prevents leakage" | Protocol diagnostics all PASS | LOW | ✅ ACCEPT |
| I4 | "first comprehensive comparison of sequence-modeling backbones... under identical leakage-controlled conditions" | 4 models compared, 3 seeds each | **HIGH** | 🔴 "identical" is false — data scale varies (full vs subset) |

**Reviewer Comment:** Line 86 claims "under identical leakage-controlled conditions" — but Table 1 mixes full-data and subset results with different epoch counts. The leakage-controlled conditions ARE identical, but the TRAINING conditions are not. Clarify.

### 1.3 Methods (lines 148-150)

| # | Claim Text | Evidence | Risk | Verdict |
|---|-----------|----------|------|---------|
| M1 | "All deep learning models... were trained under identical conditions: AdamW, lr=1e-3, batch size 64" | Fair audit experiments verify this | LOW | ✅ ACCEPT |
| M2 | "up to 20 epochs with early stopping" | Fair audit used 10 epochs for consistency | LOW | ⚠ Self-contradiction: says "up to 20" but fair comparison used 10 |

**Reviewer Comment:** Line 148 claims "up to 20 epochs" but the fair comparison (Table 1) used 10 epochs. Minor inconsistency but should be resolved.

### 1.4 Results — Architecture Comparison (lines 215-219)

| # | Claim Text | Evidence | Risk | Verdict |
|---|-----------|----------|------|---------|
| R1 | "five sequence-modeling architectures were compared at horizon 1" | Only 4 shown (MLP, LSTM, Mamba, Mamba-LSTM); Transformer missing | **HIGH** | 🔴 Missing model |
| R2 | "Mamba-LSTM model... achieved Macro F1 0.9268" | Full data, 3 epochs, single seed | MEDIUM | ✅ Accept if noted as single-run |
| R3 | "representing a substantial improvement over classical baselines" | True: 0.9268 > 0.8796 | LOW | ✅ ACCEPT |
| R4 | "MLP sequence model with mean-pooling achieved F1 0.9340, providing the strongest competition" | True | LOW | ✅ ACCEPT — honest |
| R5 | "We note that training conditions varied (MLP: 30 epochs... Mamba-LSTM: 3 epochs)" | Truthful disclosure | LOW | ✅ ACCEPT — commended |

**Reviewer Comment:** Line 215 says "five sequence-modeling architectures" but the Transformer is missing from Table 1. Either remove "five" or add a footnote explaining Transformer is excluded.

### 1.5 Results — Sequence Dependency (lines 250-252)

| # | Claim Text | Evidence | Risk | Verdict |
|---|-----------|----------|------|---------|
| S1 | "shuffled sequences achieved comparable or slightly higher performance (F1 0.9331 vs 0.9220)" | Fair audit: 3 seeds, 10 epochs | LOW | ✅ ACCEPT — honest |
| S2 | "per-cell feature expression dominates over temporal ordering" | Supported by data | LOW | ✅ ACCEPT |
| S3 | "trajectory-aware benefits may emerge at longer sequence lengths" | Speculation, not tested | MEDIUM | ⚠ Speculative — acceptable in Discussion |
| S4 | "MLP model using simple mean-pooling achieves the highest overall F1" | True | LOW | ✅ ACCEPT |

**Reviewer Comment:** The sequence order result is the most impressive finding — not because it supports the method, but because the authors report it honestly despite it undermining their "trajectory-aware" framing. This scientific integrity should be commended.

### 1.6 Discussion (lines 312-326)

| # | Claim Text | Evidence | Risk | Verdict |
|---|-----------|----------|------|---------|
| D1 | "Mamba-LSTM architecture can effectively model pseudotime-ordered multimodal single-cell sequences" | "can" qualifier + experiment support | LOW | ✅ ACCEPT |
| D2 | "selective state-space mechanism provides linear-time sequence processing" | True: O(L) complexity | LOW | ✅ ACCEPT |
| D3 | "LSTM component adds local transition refinement" | Architecture description, not performance claim | LOW | ✅ ACCEPT |
| D4 | "RNA features provided stronger classification signal than protein features" | LR RNA-only F1 0.8823 > Protein-only 0.7593 | LOW | ✅ ACCEPT |
| D5 | Six limitations — all honest | All accurate | LOW | ✅ ACCEPT |

**Reviewer Comment:** The limitations section is comprehensive and honest. I would like to see one more limitation added: "The current evidence does not establish a performance advantage for temporal ordering over mean-pooling at L=32."

### 1.7 Conclusion (lines 328-330)

| # | Claim Text | Evidence | Risk | Verdict |
|---|-----------|----------|------|---------|
| C1 | "trajectory-aware Mamba-LSTM architecture" | Architecture exists, "trajectory-aware" = pseudotime-ordered input | LOW | ✅ ACCEPT |
| C2 | "selective state-space sequence modeling with LSTM-based local refinement" | Architecture description | LOW | ✅ ACCEPT |
| C3 | "achieves strong classification performance" | "strong" relative to classical baselines | MEDIUM | ⚠ "strong" is subjective but defensible |
| C4 | "exceeding classical baselines" | Verified: +0.0482 over LR | LOW | ✅ ACCEPT |
| C5 | "establishes a reproducible foundation" | Code + data available | LOW | ✅ ACCEPT |

**Reviewer Comment:** The conclusion is appropriately restrained. No overclaims detected.

---

## 2. KEYWORD OVERCLAIM SCAN

| Keyword | Count | Context | Verdict |
|---------|-------|---------|---------|
| "outperform" | 0 | Not used | ✅ Safe |
| "superior" | 0 | Not used | ✅ Safe |
| "state-of-the-art" | 0 | Not used | ✅ Safe |
| "best" | 1 | "best individual component" (line 220) — refers to LSTM-only being the best among ablated variants | ✅ Acceptable |
| "improve" | 0 | Not used as claim | ✅ Safe |
| "demonstrate" | 3 | "demonstrates that all deep models exceed baselines" (line 65), "demonstrating the value" (line 217), "demonstrate" (line 314) | ⚠ Line 65 borderline — "demonstrates" with mixed data scales |
| "prove" | 0 | Not used | ✅ Safe |
| "novel" | 0 | Not used | ✅ Safe |
| "effective" | 2 | "effectively model" (line 314), "effective integration" (line 220) | ✅ "can effectively" is qualified |

---

## 3. MAMBA CONTRIBUTION AUDIT

### Current Language

| Location | Text | Problem? | Fix |
|----------|------|----------|-----|
| Abstract (65) | "Mamba selective state-space block for long-range sequence dependencies" | ✅ Descriptive, no performance claim | None |
| Methods (120) | "advantageous for trajectory data where state transitions are localized" | ⚠ "advantageous" implies benefit not proven | Change to "potentially advantageous" or "designed to be advantageous" |
| Results (219) | "Mamba-only model captures long-range selective dependencies" | ✅ Descriptive | None |
| Discussion (314) | "selective state-space mechanism provides linear-time sequence processing" | ✅ Factual | None |
| Discussion (314) | "LSTM component adds local transition refinement that complements Mamba's long-range selectivity" | ⚠ "complements" implies proven synergy | Change to "is designed to complement" |

### Verdict

Mamba contribution language is mostly clean. Two borderline cases:
1. Line 120: "advantageous" → "designed to be advantageous"
2. Line 314: "complements" → "is designed to complement"

These are minor but should be fixed for precision.

---

## 4. SEQUENCE VALIDITY AUDIT

### Critical Finding: Original < Shuffled (F1 0.9220 vs 0.9331)

**Audit Question:** Does the manuscript anywhere claim that trajectory ordering improves prediction?

| Location | Text Check | Verdict |
|----------|-----------|---------|
| Abstract | No claim of temporal benefit | ✅ Clean |
| Introduction | "trajectory-aware" used as descriptor, not performance claim | ✅ Clean |
| Results 3.4 | Explicitly states "shuffled sequences achieved comparable or slightly higher performance" | ✅ Honest |
| Discussion | "effectively model pseudotime-ordered... sequences" — "ordered" describes input format | ✅ Clean |
| Conclusion | "trajectory-aware" descriptor only | ✅ Clean |

**Verdict:** The manuscript correctly avoids claiming that temporal ordering improves prediction. The negative sequence-ablation result is reported transparently.

---

## 5. DATASET SCALE AUDIT

### Problem: Mixed Data Scales Across Tables

| Table | Model | Data Scale | Epochs | Seeds |
|-------|-------|-----------|--------|-------|
| Tab:baselines | LR, RF, sklearn MLP | Full (161k seq) | N/A | 3 × 4H = 12 |
| Tab:backbone | MLP | Full (161k) | 30 | 1 |
| Tab:backbone | LSTM, Mamba | Subset (10k) | 5 | 1 |
| Tab:backbone | Mamba-LSTM | Full (161k) | 3 | 1 |
| Tab:backbone (fair) | All 4 models | Subset (8k) | 10 | 3 |

**Two separate backbone tables exist with different numbers.** The text (line 217) references Table 1 with the mixed-scale single-seed numbers, but the fair comparison table (lines 222-241) supersedes it. The narrative text still references the OLD mixed-scale numbers.

**🔴 MUST FIX:** The narrative text in "Mamba-LSTM architecture comparison" (lines 215-219) references the old mixed-scale results (F1 0.9268, 0.9340, 0.9254, 0.9054) while the actual Table 1 (lines 222-241) now shows the fair comparison results (F1 0.9278, 0.9349, 0.9232, 0.9275). **The narrative and table are inconsistent.**

---

## 6. TorchSelectiveSSM AUDIT

### Check: No false claims about mamba_ssm

| Claim | Present? | Evidence |
|-------|----------|----------|
| "official Mamba implementation" | ❌ Not found | ✅ |
| "CUDA kernel" | ❌ Not found | ✅ |
| "mamba_ssm" used as description | ✅ Found but correctly contextualized | Referenced only as "Linux-only native mamba_ssm kernels" |
| "native Mamba" claimed | ❌ Not found | ✅ |
| "pure PyTorch selective SSM" | ✅ Accurate | TorchSelectiveSSM verified |

**Verdict:** ✅ CLEAN. The TorchSelectiveSSM disclosure is appropriately handled throughout.

---

## 7. BIOINFORMATICS FORMAT AUDIT

| Requirement | Status | Issue |
|-------------|--------|-------|
| **Abstract** | 198 words | ✅ Under 200-word limit |
| **Keywords** | ❌ MISSING | 🔴 **Bioinformatics requires up to 5 keywords after abstract** |
| **Introduction** | Section 1, contextualizes problem | ✅ |
| **Methods subsections** | Dataset, Split, Preprocessing, Sequence, Representation, Training, Label audit | ✅ |
| **Results subsections** | Protocol, Baselines, Architecture, Sequence, Ablation, Generalization | ✅ |
| **Discussion subsections** | Mamba-LSTM, Modality, Leakage, Limitations | ✅ |
| **Data availability** | Section present | ✅ |
| **Code availability** | GitHub link | ✅ |
| **Supplementary** | File exists but **OUTDATED** | 🔴 **Supplementary references old "evaluation framework" framing** |
| **Author contributions** | Present | ✅ |
| **Funding** | Present | ✅ |
| **Conflict of interest** | Present | ✅ |
| **References** | `references.bib` — pruned to cited entries | ✅ |

---

## 8. FIGURE AUDIT

| Figure | File | Resolution | Content Check | Verdict |
|--------|------|-----------|---------------|---------|
| **Fig 1** (framework) | `figures/fig1.pdf` | Present | Architecture overview — schematic | ⚠ Old figure; may not match updated architecture description |
| **Fig 2** (workflow) | `figures/fig2.pdf` | Present | Donor-held-out workflow | ⚠ Old figure; "evaluation workflow" framing |
| **Fig 3** (backbone) | `figures/fig3_backbone_comparison.pdf` | PDF+SVG+PNG 600dpi ✅ | Backbone comparison bar chart | ✅ Generated from real data |
| **Fig 4** (ablation) | `figures/fig4_ablation.pdf` | PDF+SVG+PNG 600dpi ✅ | Architecture + modality ablation | ✅ Generated from real data |

**🔴 CRITICAL:** Figures 1 and 2 are from the OLD "evaluation framework" version and reference content ("evaluation workflow") that no longer matches the updated "method paper" framing. The captions in main.tex have been updated but the figure CONTENT still shows old framing.

---

## 9. SUPPLEMENTARY AUDIT

### Current supplementary.tex (lines 38-43)

```
\subsubsection*{SM3. Native Mamba status}
The scLifeMamba architecture specifies a Mamba-LSTM sequence module, but the
leakage-safe rerun environment was Windows CPU-only. CUDA, mamba_ssm, and
causal_conv1d were unavailable. Therefore, native-Mamba model results are
not reported as completed formal results in the main manuscript.

\subsubsection*{SM4. Archived invalid results}
...Those values are not used as formal evidence in the revised manuscript.
```

**🔴 CRITICAL:** SM3 and SM4 are **DIRECTLY CONTRADICTED** by the updated main.tex which:
1. Reports Mamba-LSTM results (F1 0.9278 ± 0.0053)
2. Uses TorchSelectiveSSM as the validated backend (CUDA, not CPU-only)
3. Reports fair comparison experiments as formal results

**The supplementary describes a paper that no longer exists.** This must be completely rewritten to match the current manuscript.

---

## 10. GITHUB AUDIT

**Repository:** https://github.com/TheFinalDreamer/scLifeMamba
**Latest commit:** `eec459a` ("v2.0: Mamba-LSTM method paper refactoring")

| Check | Status | Issue |
|-------|--------|-------|
| README architecture description | ✅ Updated | Matches paper |
| README results table | ✅ Updated | Honest numbers |
| "SOTA" / "best" / "outperform" | ✅ NOT FOUND | Clean |
| Installation instructions | ✅ Present | |
| Dataset preparation | ✅ Present | |
| Training command | ✅ Present | |
| Evaluation command | ✅ Present | |
| Reproduce section | ✅ Present | |
| Citation format | ✅ Present | |
| LICENSE | ✅ MIT | |

**Verdict:** ✅ README is consistent with paper. No overclaims found.

---

## 11. CRITICAL INCONSISTENCIES SUMMARY

### 🔴 BLOCKING (Must Fix Before Submission)

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| **B1** | **Narrative text references OLD mixed-scale numbers while Table 1 shows NEW fair comparison numbers** | Lines 215-219 vs Table back:bone | Rewrite §3.3 to reference ONLY the fair comparison table |
| **B2** | **Supplementary SM3/SM4 contradicts main manuscript** | supplementary.tex lines 38-52 | Delete SM3/SM4; replace with updated content |
| **B3** | **Abstract mixes full-data, single-seed, and subset results without distinguishing** | Lines 65-66 | Either: (a) use only fair comparison numbers, or (b) add explicit data-scale qualifiers |
| **B4** | **Transformer mentioned in Introduction (line 86) and Methods (line 148) but MISSING from all results tables** | Lines 86, 148, Table back:bone | Either add Transformer results or remove mention |

### 🟡 WARNING (Should Fix, Not Blocking)

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| W1 | "Keywords" missing (Bioinformatics format requirement) | After abstract | Add `\textbf{Keywords:}` |
| W2 | Fig 1 and Fig 2 are old evaluation framework figures | fig1.pdf, fig2.pdf | Regenerate with method paper framing (or update captions to match old figure content) |
| W3 | "advantageous" (line 120) overstates Mamba benefit | Methods §2.5.1 | "designed to be advantageous" |
| W4 | "complements" (line 314) implies proven synergy | Discussion §4.1 | "is designed to complement" |
| W5 | "five... architectures" counts Transformer which isn't shown | Line 215 | "four sequence-modeling architectures" |
| W6 | Methods says "up to 20 epochs"; fair comparison used 10 | Line 148 | Harmonize to 10 or note discrepancy |
| W7 | MLP method name collision: "MLP (mean-pool)" vs "MLP (sklearn)" | Tables | Rename for clarity: "MLP-Seq (mean-pool)" vs "MLP-Classic (sklearn)" |

---

## 12. FINAL DECISION

### VERDICT: MINOR REVISION REQUIRED → READY_FOR_SUBMISSION AFTER FIXES

This manuscript has been transformed from a cautious evaluation framework paper into an honest method paper. The scientific integrity displayed — particularly the transparent reporting of the negative sequence-order result and the honest acknowledgment that MLP outperforms Mamba-LSTM — is commendable and rare.

**Required fixes (estimated 2-4 hours):**
1. 🔴 Rewrite §3.3 narrative to match fair comparison table (B1)
2. 🔴 Update supplementary.tex SM3/SM4 (B2)
3. 🔴 Harmonize abstract numbers to fair comparison or add qualifiers (B3)
4. 🔴 Add Transformer note or remove mention (B4)
5. 🟡 Add Keywords (W1)
6. 🟡 Fix 5 minor wording issues (W2-W7)

**After these fixes, the manuscript is ready for Bioinformatics submission.**

The core contribution — Mamba-LSTM architecture for multimodal single-cell sequence modeling with honest evaluation — is valid and well-supported. The paper's greatest strength is what it does NOT claim: no false superiority, no hidden data scales, no suppressed negative results.

---

**Simulated Reviewer Signature:**
*Bioinformatics Adversarial Reviewer #1*
*Recommendation: Minor Revision → Accept*
