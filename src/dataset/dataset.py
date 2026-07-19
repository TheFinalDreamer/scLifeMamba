"""SingleCellMultiModalDataset supporting synthetic, h5ad, and processed modes."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse


def synthetic_data_generator(
    num_cells: int = 2000,
    rna_dim: int = 2000,
    protein_dim: int = 30,
    num_classes: int = 5,
    seed: int = 42,
):
    """Generate synthetic multi-modal single-cell data for testing.

    Returns:
        x_rna, x_protein, labels, pseudotime, adjacency
    """
    rng = np.random.default_rng(seed)

    pseudotime = rng.uniform(0, 1, size=num_cells)
    pseudotime.sort()
    pseudotime = pseudotime.reshape(-1)

    interval_boundaries = np.linspace(0, 1, num_classes + 1)
    labels = np.digitize(pseudotime, interval_boundaries[1:-1])

    rna_means = rng.normal(0, 1, size=(num_classes, rna_dim))
    x_rna = np.zeros((num_cells, rna_dim))
    for i in range(num_cells):
        cls = labels[i]
        x_rna[i] = rng.normal(rna_means[cls], 0.5, size=rna_dim) + 0.3 * pseudotime[i]
    x_rna = np.maximum(x_rna, 0)

    projection = rng.normal(0, 0.1, size=(rna_dim, protein_dim))
    x_protein = x_rna @ projection + 0.5 * pseudotime.reshape(-1, 1)
    x_protein += rng.normal(0, 0.1, size=x_protein.shape)
    x_protein = np.maximum(x_protein, 0)

    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)
    for i in range(num_cells):
        diff = np.abs(pseudotime - pseudotime[i])
        neighbors = np.argsort(diff)[1:6]
        adjacency[i, neighbors] = np.exp(-diff[neighbors] * 5)
        adjacency[neighbors, i] = adjacency[i, neighbors]

    d = adjacency.sum(axis=1, keepdims=True)
    d[d == 0] = 1
    adjacency = adjacency / d

    return (x_rna.astype(np.float32), x_protein.astype(np.float32),
            labels.astype(np.int64), pseudotime.astype(np.float32),
            adjacency.astype(np.float32))


def _load_h5ad_data(config) -> dict:
    """Load and process data from an AnnData h5ad file.

    Returns dict with keys: x_rna, x_protein, labels, pseudotime, adjacency,
    label_mapping, batch, use_protein, use_pseudotime
    """
    try:
        import anndata
    except ImportError:
        raise ImportError("anndata is required for h5ad mode. Install with: pip install anndata scanpy")

    cfg = config
    path = cfg.h5ad_path
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"h5ad file not found: {path}")

    adata = anndata.read_h5ad(path)

    # Subsample if max_cells is set
    max_cells = cfg.get("max_cells", None)
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[idx].copy()

    # RNA: use layer or X
    rna_layer = cfg.get("rna_layer", None)
    if rna_layer is not None and rna_layer in adata.layers:
        x_rna = adata.layers[rna_layer]
    else:
        x_rna = adata.X

    if issparse(x_rna):
        x_rna = x_rna.toarray()
    x_rna = np.asarray(x_rna, dtype=np.float32)

    # Protein
    use_protein = cfg.get("use_protein", True)
    protein_key = cfg.get("protein_obsm_key", "protein")
    if use_protein:
        if protein_key in adata.obsm:
            x_protein = np.asarray(adata.obsm[protein_key], dtype=np.float32)
        else:
            raise KeyError(
                f"use_protein=True but '{protein_key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}. "
                f"Set use_protein=False for RNA-only mode."
            )
    else:
        # RNA-only: create zero placeholder
        x_protein = np.zeros((x_rna.shape[0], 1), dtype=np.float32)

    # Labels
    label_key = cfg.get("label_key", "cell_type")
    if label_key not in adata.obs:
        raise KeyError(f"label_key '{label_key}' not found in adata.obs. Available: {list(adata.obs.columns)}")

    label_col = adata.obs[label_key]
    if label_col.dtype.name == "category":
        label_categories = list(label_col.cat.categories)
        labels = label_col.cat.codes.values.astype(np.int64)
        label_mapping = {i: str(c) for i, c in enumerate(label_categories)}
    else:
        # Encode non-categorical labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(label_col.values.astype(str)).astype(np.int64)
        label_mapping = {i: str(c) for i, c in enumerate(le.classes_)}

    # Pseudotime (optional)
    use_pseudotime = cfg.get("use_pseudotime", False)
    ptime_key = cfg.get("pseudotime_key", "pseudotime")
    pseudotime = None
    if use_pseudotime:
        if ptime_key in adata.obs:
            pseudotime = adata.obs[ptime_key].values.astype(np.float32)
            # Normalize to [0, 1] if not already
            pmin, pmax = pseudotime.min(), pseudotime.max()
            if pmax > pmin:
                pseudotime = (pseudotime - pmin) / (pmax - pmin)
        else:
            print(f"Warning: use_pseudotime=True but '{ptime_key}' not found in adata.obs. Disabling pseudotime.")
            use_pseudotime = False

    # Batch (optional)
    batch_key = cfg.get("batch_key", None)
    batch = None
    if batch_key is not None and batch_key in adata.obs:
        batch_col = adata.obs[batch_key]
        if batch_col.dtype.name == "category":
            batch = batch_col.cat.codes.values.astype(np.int64)
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            batch = le.fit_transform(batch_col.values.astype(str)).astype(np.int64)

    # Adjacency (optional, from kNN graph if available)
    adjacency = None
    if "neighbors" in adata.uns:
        try:
            from scipy.sparse import issparse as _issparse
            neighbors = adata.obsp.get("connectivities", None)
            if neighbors is not None:
                if _issparse(neighbors):
                    adjacency = neighbors.toarray().astype(np.float32)
                else:
                    adjacency = np.asarray(neighbors, dtype=np.float32)
        except Exception:
            pass

    # Normalization (optional — applied to RNA)
    normalize = cfg.get("normalize", False)
    log1p = cfg.get("log1p", False)
    scale = cfg.get("scale", False)

    if normalize:
        from .preprocessing import normalize_counts
        x_rna = normalize_counts(x_rna)
        if use_protein and x_protein.shape[1] > 1:
            x_protein = normalize_counts(x_protein)

    if log1p:
        x_rna = np.log1p(x_rna)
        if use_protein and x_protein.shape[1] > 1:
            x_protein = np.log1p(x_protein)

    if scale:
        from .preprocessing import scale_data
        x_rna = scale_data(x_rna)

    return {
        "x_rna": x_rna,
        "x_protein": x_protein,
        "labels": labels,
        "pseudotime": pseudotime,
        "adjacency": adjacency,
        "label_mapping": label_mapping,
        "batch": batch,
        "use_protein": use_protein,
        "use_pseudotime": use_pseudotime,
    }


def _load_processed_data(config) -> dict:
    """Load preprocessed data from cache directory.

    Expects processed_dir with: x_rna.npy/npz, x_protein.npy/npz, labels.npy/npz,
    label_mapping.json, split_indices.json, pseudotime.npy/npz (optional),
    batch.npy/npz (optional), data_summary.json (optional)
    """
    d = config.processed_dir
    if not d or not os.path.isdir(d):
        raise FileNotFoundError(f"processed_dir not found: {d}")

    def _load_array(name):
        npy_path = os.path.join(d, f"{name}.npy")
        npz_path = os.path.join(d, f"{name}.npz")
        if os.path.exists(npy_path):
            return np.load(npy_path)
        elif os.path.exists(npz_path):
            return np.load(npz_path)["arr_0"]
        return None

    x_rna = _load_array("x_rna")
    if x_rna is None:
        raise FileNotFoundError(f"x_rna.npy/npz not found in {d}")

    x_protein = _load_array("x_protein")
    labels = _load_array("labels")
    pseudotime = _load_array("pseudotime")
    batch = _load_array("batch")

    label_mapping = {}
    lm_path = os.path.join(d, "label_mapping.json")
    if os.path.exists(lm_path):
        with open(lm_path, "r") as f:
            label_mapping = json.load(f)

    use_protein = x_protein is not None and x_protein.shape[1] > 1
    use_pseudotime = pseudotime is not None

    if x_protein is None:
        x_protein = np.zeros((x_rna.shape[0], 1), dtype=np.float32)

    return {
        "x_rna": x_rna.astype(np.float32),
        "x_protein": x_protein.astype(np.float32),
        "labels": labels.astype(np.int64) if labels is not None else None,
        "pseudotime": pseudotime.astype(np.float32) if pseudotime is not None else None,
        "adjacency": None,
        "label_mapping": label_mapping,
        "batch": batch.astype(np.int64) if batch is not None else None,
        "use_protein": use_protein,
        "use_pseudotime": use_pseudotime,
    }


class SingleCellMultiModalDataset(Dataset):
    """Dataset for single-cell multi-modal data.

    Supports three modes:
    - synthetic: auto-generate toy data for testing
    - h5ad: read from AnnData file with preprocessing
    - processed: read preprocessed cache files
    """

    def __init__(self, data: dict, mode: str = "synthetic"):
        self.mode = mode
        self.x_rna = torch.as_tensor(data["x_rna"], dtype=torch.float32)
        self.x_protein = torch.as_tensor(data["x_protein"], dtype=torch.float32)

        self.has_labels = data.get("labels") is not None
        self.labels = torch.as_tensor(data["labels"], dtype=torch.long) if self.has_labels else None

        self.has_pseudotime = data.get("pseudotime") is not None
        self.pseudotime = torch.as_tensor(data["pseudotime"], dtype=torch.float32) if self.has_pseudotime else None

        self.has_adjacency = data.get("adjacency") is not None
        self.adjacency = torch.as_tensor(data["adjacency"], dtype=torch.float32) if self.has_adjacency else None

        # Metadata
        self.label_mapping = data.get("label_mapping", {})
        self.use_protein = data.get("use_protein", self.x_protein.shape[1] > 1)
        self.use_pseudotime = data.get("use_pseudotime", self.has_pseudotime)

    def __len__(self):
        return len(self.x_rna)

    def __getitem__(self, idx):
        item = {
            "x_rna": self.x_rna[idx],
            "x_protein": self.x_protein[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        if self.pseudotime is not None:
            item["pseudotime"] = self.pseudotime[idx]
        return item

    def get_adjacency(self):
        if self.has_adjacency and self.adjacency is not None:
            return self.adjacency
        return None
