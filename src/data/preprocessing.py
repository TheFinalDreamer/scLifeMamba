"""Preprocessing utilities for single-cell data."""

import numpy as np
from scipy.sparse import issparse


def normalize_counts(x: np.ndarray, target_sum: float = 10000.0) -> np.ndarray:
    """Library-size normalization: normalize each cell to target_sum counts."""
    if issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    lib_size = x.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1
    return (x / lib_size * target_sum).astype(np.float32)


def log1p_normalize(x: np.ndarray) -> np.ndarray:
    """Log1p transform."""
    return np.log1p(np.asarray(x, dtype=np.float32))


def scale_data(x: np.ndarray) -> np.ndarray:
    """Standard scale to zero mean and unit variance per gene."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return ((x - mean) / std).astype(np.float32)


def select_hvg(x: np.ndarray, top_k: int = 2000) -> np.ndarray:
    """Select highly variable genes by variance.

    Args:
        x: (n_cells, n_genes) expression matrix
        top_k: number of top variable genes to keep

    Returns:
        Filtered array with top_k genes
    """
    if top_k is None or top_k >= x.shape[1]:
        return x
    x = np.asarray(x, dtype=np.float64)
    variances = np.var(x, axis=0)
    top_indices = np.argsort(variances)[-top_k:]
    return x[:, top_indices].astype(np.float32)


def filter_genes(x: np.ndarray, min_cells: int = 3) -> np.ndarray:
    """Filter genes expressed in fewer than min_cells.

    Returns unchanged array for now — full implementation depends on count data.
    """
    return x


def encode_labels(labels) -> tuple:
    """Encode string/categorical labels to integer codes.

    Args:
        labels: array-like of labels (can be strings, ints, or categorical)

    Returns:
        (encoded_labels: np.ndarray[int64], label_mapping: dict)
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    encoded = le.fit_transform(np.asarray(labels, dtype=str))
    mapping = {i: str(c) for i, c in enumerate(le.classes_)}
    return encoded.astype(np.int64), mapping


def preprocess_rna(x_rna: np.ndarray,
                   normalize: bool = True,
                   log1p: bool = True,
                   hvg_top_k: int = None) -> np.ndarray:
    """Standard RNA preprocessing pipeline: normalize + log1p + optional HVG."""
    if normalize:
        x_rna = normalize_counts(x_rna)
    if log1p:
        x_rna = log1p_normalize(x_rna)
    if hvg_top_k is not None and hvg_top_k < x_rna.shape[1]:
        x_rna = select_hvg(x_rna, top_k=hvg_top_k)
    return x_rna


def preprocess_protein(x_protein: np.ndarray,
                       normalize: bool = True,
                       log1p: bool = True) -> np.ndarray:
    """Standard protein preprocessing: normalize + log1p."""
    if normalize:
        x_protein = normalize_counts(x_protein)
    if log1p:
        x_protein = log1p_normalize(x_protein)
    return x_protein


def build_data_summary(x_rna, x_protein, labels, pseudotime, label_mapping,
                       batch=None, use_protein=True, use_pseudotime=False) -> dict:
    """Build a data summary dict for documentation and reproducibility."""
    summary = {
        "n_cells": int(x_rna.shape[0]),
        "rna_dim": int(x_rna.shape[1]),
        "protein_dim": int(x_protein.shape[1]) if x_protein is not None else 0,
        "n_classes": int(len(np.unique(labels))) if labels is not None else 0,
        "use_protein": bool(use_protein),
        "use_pseudotime": bool(use_pseudotime),
        "label_mapping": label_mapping,
        "rna_mean": float(np.mean(x_rna)),
        "rna_std": float(np.std(x_rna)),
    }
    if x_protein is not None and x_protein.shape[1] > 1:
        summary["protein_mean"] = float(np.mean(x_protein))
        summary["protein_std"] = float(np.std(x_protein))
    if pseudotime is not None:
        summary["pseudotime_min"] = float(np.min(pseudotime))
        summary["pseudotime_max"] = float(np.max(pseudotime))
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        summary["class_distribution"] = {str(label_mapping.get(k, k)): int(c) for k, c in zip(unique, counts)}
    if batch is not None:
        summary["n_batches"] = int(len(np.unique(batch)))
    return summary
