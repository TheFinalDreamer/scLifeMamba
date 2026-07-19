"""Train/val/test splitting utilities."""

import numpy as np
import json
import os


def stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Stratified split indices into train/val/test.

    Args:
        labels: (num_cells,) integer class labels
        train_ratio, val_ratio, test_ratio: split proportions
        seed: random seed

    Returns:
        train_indices, val_indices, test_indices (np.ndarrays)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    num_cells = len(labels)
    indices = np.arange(num_cells)

    train_idx, val_idx, test_idx = [], [], []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = indices[labels == label]
        rng.shuffle(label_indices)
        n = len(label_indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_idx.append(label_indices[:n_train])
        val_idx.append(label_indices[n_train:n_train + n_val])
        test_idx.append(label_indices[n_train + n_val:])

    train_indices = np.sort(np.concatenate(train_idx))
    val_indices = np.sort(np.concatenate(val_idx))
    test_indices = np.sort(np.concatenate(test_idx))

    return train_indices, val_indices, test_indices


def save_split_indices(train_idx, val_idx, test_idx, filepath: str):
    """Save split indices to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        }, f, indent=2)
