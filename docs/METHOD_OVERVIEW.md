# Method overview

## Architecture

scMultiLifeMamba encodes pseudotime-ordered multi-omics trajectory windows using:

1. **Modality-specific encoders**: RNA encoder (MLP/Transformer), protein encoder (MLP), optional ATAC encoder.
2. **LagAwareDynamicFusion**: Pseudotime- and horizon-conditioned gating network producing adaptive modality weights.
3. **Mamba-LSTM encoder**: Gated fusion of Mamba SSM output and bidirectional LSTM output for sequence encoding.
4. **Task heads**: Classification (lifecycle stage), regression (pseudotime), branch prediction, embedding.

## Mamba-LSTM Encoder

```
Input: (B, L, d_model) trajectory sequence
  → MambaBlock (selective SSM scanning)
  → LSTM (bidirectional recurrence)
  → h_mamba + gamma * h_lstm  (learned gate)
  → center-pooling
  → LayerNorm + Dropout
Output: (B, d_model) pooled representation
```

## LagAwareDynamicFusion

```
Modality encodings: z_rna, z_protein, [z_atac]
  + Pseudotime embedding (learnable bins)
  + Horizon embedding
  + Task embedding
  → Gate MLP
  → softmax over modalities
  → weighted sum: z_fused = sum(alpha_i * z_i)
```

## Tasks

| Task | Head | Loss |
|------|------|------|
| Lifecycle stage prediction | ClassificationHead | Cross-entropy |
| Future pseudotime regression | RegressionHead | MSE |
| Trajectory direction prediction | ClassificationHead | Cross-entropy |

## Modality dominance analysis

The framework supports systematic comparison of RNA-only, protein-only, and fused modality configurations under identical model architecture. This enables quantification of modality contributions to lifecycle phenotype prediction.
