"""
LagAwareDynamicFusion: pseudotime-aware + horizon-aware modality fusion.

Design:
    alpha = softmax(Gate([z_rna, z_protein, pseudotime_embed, horizon_embed, task_embed]))
    z_fused = alpha_rna * z_rna + alpha_protein * z_protein (+ alpha_atac * z_atac)

Supports:
  - 2D input [B, D] — automatically treated as single-timestep
  - 3D input [B, L, D] — preserves sequence dimension
  - pseudotime: [B], [B, L], or scalar
  - horizon: scalar, [B], or [B, L]
  - ATAC modality (optional)

Saves modality_weights tracking alpha values over pseudotime bins.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LagAwareDynamicFusion(nn.Module):
    """Dynamic modality fusion with pseudotime and horizon embeddings.

    Args:
        d_model: Hidden dimension for all projections
        num_modalities: 2 (RNA+Protein) or 3 (RNA+ATAC+Protein)
        n_pseudotime_bins: Number of pseudotime bins for discretization (default 20)
        max_horizon: Maximum prediction horizon (default 16)
        dropout: Dropout rate for gate network
        use_atac: Whether ATAC modality is used
    """

    def __init__(
        self,
        d_model: int = 128,
        num_modalities: int = 2,
        rna_dim: int = None,  # kept for backward compat
        protein_dim: int = None,  # kept for backward compat
        atac_dim: int = None,  # kept for backward compat
        hidden_dim: int = None,  # kept for backward compat
        n_pseudotime_bins: int = 20,
        max_horizon: int = 16,
        use_atac: bool = False,
        use_protein: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Support both old (rna_dim, protein_dim, hidden_dim) and new (d_model, num_modalities) API
        if hidden_dim is not None:
            d_model = hidden_dim
        if rna_dim is not None:
            d_model = rna_dim if hidden_dim is None else hidden_dim
        if protein_dim is not None and num_modalities == 2:
            pass  # protein_dim used for backward compat only

        self.d_model = d_model
        self.num_modalities = num_modalities
        self.use_atac = use_atac
        self.use_protein = use_protein
        self.max_horizon = max_horizon

        if use_atac:
            self.num_modalities = 3
        if not use_protein and not use_atac:
            self.num_modalities = 1

        # Modality projections
        self.rna_proj = nn.Linear(d_model, d_model)
        self.atac_proj = nn.Linear(d_model, d_model) if use_atac else None
        self.protein_proj = nn.Linear(d_model, d_model) if use_protein else None

        # Pseudotime embedding (learnable bins)
        self.n_pseudotime_bins = n_pseudotime_bins
        self.pseudotime_embed = nn.Embedding(n_pseudotime_bins, d_model // 4)

        # Horizon embedding
        self.horizon_embed = nn.Embedding(max_horizon + 1, d_model // 4)

        # Task embedding (optional, for multi-task scenarios)
        self.task_embed = nn.Embedding(4, d_model // 4)  # rna->prot, prot->rna, joint->state, joint->ptime

        # Gate network: takes modality encodings + pseudotime + horizon + task → weights
        gate_input_dim = d_model * self.num_modalities + (d_model // 4) * 3
        if self.num_modalities > 1:
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, self.num_modalities),
            )
        else:
            self.gate = None

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Track weights for analysis
        self.register_buffer("_weight_buffer", torch.zeros(0))
        self._last_alpha = None  # for debugging

    def _get_pseudotime_bin(self, pseudotime: torch.Tensor) -> torch.LongTensor:
        """Discretize pseudotime into bins [0, n_pseudotime_bins-1]."""
        if pseudotime.dim() > 1:
            pseudotime = pseudotime.reshape(-1)
        p_min = pseudotime.min()
        p_max = pseudotime.max()
        if p_max - p_min < 1e-8:
            normalized = torch.zeros_like(pseudotime)
        else:
            normalized = (pseudotime - p_min) / (p_max - p_min)
        bins = (normalized * (self.n_pseudotime_bins - 1)).long().clamp(0, self.n_pseudotime_bins - 1)
        return bins

    def _ensure_2d(self, t, B):
        """Ensure tensor is [B, D] format."""
        if t is None:
            return None
        if t.dim() == 3:
            # [B, L, D] -> pool to [B, D] via mean
            return t.mean(dim=1)
        return t

    def forward(
        self,
        z_rna: torch.Tensor,
        z_protein: torch.Tensor = None,
        z_atac: torch.Tensor = None,
        pseudotime: torch.Tensor = None,
        horizon: torch.Tensor = None,
        task_id: int = 0,
        modality_mask: torch.Tensor = None,
        return_weights: bool = True,
    ):
        """Forward pass with automatic 2D/3D handling.

        Args:
            z_rna: RNA encoding — [B, D] or [B, L, D]
            z_protein: Protein encoding — [B, D] or [B, L, D]
            z_atac: ATAC encoding — [B, D] or [B, L, D]
            pseudotime: [B], [B, L], scalar, or None
            horizon: int, [B], [B, L], or None
            task_id: int (0=rna_to_protein, 1=protein_to_rna, 2=joint_to_state, 3=joint_to_pseudotime)
            modality_mask: [B, n_modalities] boolean mask
            return_weights: Whether to return modality weights

        Returns:
            z_fused: Fused representation [B, D] or [B, L, D] (same shape as input)
            weights: dict with alpha_rna, alpha_protein, alpha_atac, pseudotime, horizon
        """
        # --- Detect input dimensionality ---
        is_3d = z_rna.dim() == 3
        if is_3d:
            B, L, Dr = z_rna.shape
            z_rna_flat = z_rna.reshape(B * L, Dr)
            z_protein_flat = None
            z_atac_flat = None
            if z_protein is not None:
                z_protein_flat = z_protein.reshape(B * L, -1) if z_protein.dim() == 3 else z_protein
            if z_atac is not None:
                z_atac_flat = z_atac.reshape(B * L, -1) if z_atac.dim() == 3 else z_atac
            # Expand pseudotime
            if pseudotime is not None:
                if pseudotime.dim() == 1:
                    pt_flat = pseudotime.unsqueeze(-1).expand(B, L).reshape(B * L)
                elif pseudotime.dim() == 2 and pseudotime.size(1) == L:
                    pt_flat = pseudotime.reshape(B * L)
                else:
                    pt_flat = pseudotime.reshape(-1)[:B * L]
            else:
                pt_flat = None
            # Expand horizon
            if horizon is not None:
                if isinstance(horizon, (int, float)):
                    h_flat = torch.full((B * L,), int(horizon), dtype=torch.long, device=z_rna.device)
                elif isinstance(horizon, torch.Tensor):
                    if horizon.dim() == 0 or horizon.numel() == 1:
                        h_flat = torch.full((B * L,), int(horizon.item()), dtype=torch.long, device=z_rna.device)
                    elif horizon.dim() == 1:
                        h_flat = horizon.unsqueeze(-1).expand(B, L).reshape(B * L)
                    else:
                        h_flat = horizon.reshape(-1)[:B * L]
                else:
                    h_flat = torch.zeros(B * L, dtype=torch.long, device=z_rna.device)
            else:
                h_flat = torch.zeros(B * L, dtype=torch.long, device=z_rna.device)
        else:
            B = z_rna.size(0)
            z_rna_flat = z_rna
            z_protein_flat = z_protein
            z_atac_flat = z_atac
            pt_flat = pseudotime
            if horizon is not None:
                if isinstance(horizon, (int, float)):
                    h_flat = torch.full((B,), int(horizon), dtype=torch.long, device=z_rna.device)
                elif isinstance(horizon, torch.Tensor):
                    h_flat = horizon.reshape(-1)[:B].long()
                else:
                    h_flat = torch.zeros(B, dtype=torch.long, device=z_rna.device)
            else:
                h_flat = torch.zeros(B, dtype=torch.long, device=z_rna.device)

        # --- Project modalities ---
        h_rna = self.rna_proj(z_rna_flat)
        components = [h_rna]

        if self.use_atac:
            if z_atac_flat is not None and self.atac_proj is not None:
                components.append(self.atac_proj(z_atac_flat))
            else:
                components.append(torch.zeros_like(h_rna))
        if self.use_protein:
            if z_protein_flat is not None and self.protein_proj is not None:
                components.append(self.protein_proj(z_protein_flat))
            else:
                components.append(torch.zeros_like(h_rna))

        Bf = z_rna_flat.size(0)

        # --- Gate network ---
        if self.gate is not None and self.num_modalities > 1:
            gate_parts = components.copy()

            # Pseudotime embedding
            if pt_flat is not None:
                p_bins = self._get_pseudotime_bin(pt_flat)
                p_embed = self.pseudotime_embed(p_bins)  # [Bf, d_model//4]
            else:
                p_embed = torch.zeros(Bf, self.d_model // 4, device=z_rna_flat.device)

            # Horizon embedding
            h_idx = h_flat.clamp(0, self.max_horizon).long()
            h_embed = self.horizon_embed(h_idx)  # [Bf, d_model//4]

            # Task embedding
            t_idx = torch.full((Bf,), task_id, dtype=torch.long, device=z_rna_flat.device)
            t_embed = self.task_embed(t_idx)  # [Bf, d_model//4]

            gate_parts.append(p_embed)
            gate_parts.append(h_embed)
            gate_parts.append(t_embed)

            gate_input = torch.cat(gate_parts, dim=-1)
            alpha = F.softmax(self.gate(gate_input), dim=-1)  # [Bf, n_modalities]

            if modality_mask is not None:
                alpha = alpha * modality_mask.float()
                alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)

            z_fused_flat = sum(alpha[:, i:i + 1] * comp for i, comp in enumerate(components))
        else:
            z_fused_flat = components[0]
            alpha = torch.ones(Bf, 1, device=z_rna_flat.device)

        z_fused_flat = self.output_proj(z_fused_flat)

        # --- Reshape back if 3D ---
        if is_3d:
            z_fused = z_fused_flat.reshape(B, L, -1)
        else:
            z_fused = z_fused_flat

        # --- Build weights dict ---
        self._last_alpha = alpha.detach()

        if return_weights:
            weights = {"alpha_rna": alpha[:, 0].detach()}
            idx = 1
            if self.use_protein and self.num_modalities > 1:
                weights["alpha_protein"] = alpha[:, idx].detach()
                idx += 1
            if self.use_atac and self.num_modalities > 1:
                weights["alpha_atac"] = alpha[:, idx].detach()
                idx += 1
            if pt_flat is not None:
                weights["pseudotime"] = pt_flat.detach()
            weights["horizon"] = h_flat.detach() if isinstance(h_flat, torch.Tensor) else torch.tensor([0])
            weights["task_id"] = task_id
            return z_fused, weights

        return z_fused, None

    def get_weight_summary(self):
        """Return summary of tracked modality weights."""
        if self._last_alpha is None:
            return None
        alpha = self._last_alpha
        result = {}
        result["alpha_rna_mean"] = alpha[:, 0].mean().item()
        if alpha.size(1) > 1:
            result["alpha_protein_mean"] = alpha[:, 1].mean().item()
        if alpha.size(1) > 2:
            result["alpha_atac_mean"] = alpha[:, 2].mean().item()
        return result
