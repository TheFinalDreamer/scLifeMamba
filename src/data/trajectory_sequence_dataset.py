"""Trajectory-aware cell-state sequence dataset for Phase 6.

Unlike Phase 5 gene-index sequences (shape: B, n_genes, 1), Phase 6 uses
trajectory-aware cell-state sequences (shape: B, seq_len, feature_dim)
where each time step is a 50-dim PCA cell state vector ordered along
DPT pseudotime.

Boundary cells use a pad-to-L strategy: the window is always exactly L=16
cells centered on the target cell along the pseudotime-ordered list.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectorySequenceDataset(Dataset):
    """Dataset of pseudotime-window cell-state sequences.

    Each item is a sequence of L cell-state vectors ordered along pseudotime.
    """

    def __init__(self, sequences, labels, pseudotime, cell_indices=None):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.pseudotime = torch.as_tensor(pseudotime, dtype=torch.float32)
        if cell_indices is None:
            cell_indices = np.arange(len(sequences))
        self.cell_indices = np.asarray(cell_indices)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "x": self.sequences[idx],
            "label": self.labels[idx],
            "pseudotime": self.pseudotime[idx],
        }


class TrajectorySequenceDataModule:
    """Data module wrapping trajectory sequence train/val/test splits.

    Loads pre-built .npy files and applies Phase 4 split indices.
    """

    def __init__(
        self,
        data_dir,
        processed_dir=None,
        batch_size=64,
        num_workers=0,
    ):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup()

    def _load(self, name):
        path = os.path.join(self.data_dir, name + ".npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def setup(self):
        """Load sequences and apply train/val/test splits."""
        sequences = self._load("sequences_pseudotime_window_L16")
        labels = self._load("labels")
        pseudotime = self._load("pseudotime")

        if sequences is None:
            raise FileNotFoundError(
                "sequences_pseudotime_window_L16.npy not found in " + self.data_dir
            )

        self.n_classes = int(labels.max()) + 1
        self.feature_dim = sequences.shape[2]
        self.seq_len = sequences.shape[1]
        n_cells = len(sequences)

        # Load split indices (from Phase 4 preprocessed data)
        split_indices = None
        if self.processed_dir:
            split_path = os.path.join(self.processed_dir, "split_indices.json")
        else:
            # Try to locate relative to data_dir
            parent = os.path.dirname(os.path.dirname(os.path.dirname(self.data_dir)))
            split_path = os.path.join(parent, "processed", "paul15", "split_indices.json")

        if os.path.exists(split_path):
            with open(split_path) as f:
                split_indices = json.load(f)

        if split_indices is not None:
            train_idx = split_indices.get("train_idx", [])
            val_idx = split_indices.get("val_idx", [])
            test_idx = split_indices.get("test_idx", [])
        else:
            # Fallback: 60/20/20 random split
            rng = np.random.RandomState(42)
            idx = rng.permutation(n_cells)
            n_train = int(n_cells * 0.6)
            n_val = int(n_cells * 0.2)
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]

        self.train_dataset = TrajectorySequenceDataset(
            sequences[train_idx], labels[train_idx], pseudotime[train_idx]
        )
        self.val_dataset = TrajectorySequenceDataset(
            sequences[val_idx], labels[val_idx], pseudotime[val_idx]
        )
        self.test_dataset = TrajectorySequenceDataset(
            sequences[test_idx], labels[test_idx], pseudotime[test_idx]
        )

        self.train_size = len(self.train_dataset)
        self.val_size = len(self.val_dataset)
        self.test_size = len(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )
