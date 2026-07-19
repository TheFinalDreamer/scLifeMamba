"""PyTorch Lightning-style DataModule for single-cell multi-modal data."""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    SingleCellMultiModalDataset,
    synthetic_data_generator,
    _load_h5ad_data,
    _load_processed_data,
)
from .split import stratified_split, save_split_indices


class MultiModalDataModule:
    """Encapsulates dataset creation, splitting, and DataLoader construction.

    Supports modes: synthetic, h5ad, processed
    """

    def __init__(self, config):
        self.config = config
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.get("num_workers", 0)
        self.seed = config.project.seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        self.adjacency = None
        self.label_mapping = {}
        self.use_protein = True
        self.use_pseudotime = False
        self.data_summary = {}

    def setup(self):
        """Create datasets based on config mode."""
        mode = self.config.data.mode

        if mode == "synthetic":
            data = self._setup_synthetic()
            load_split_indices = False
        elif mode == "h5ad":
            data = self._setup_h5ad()
            load_split_indices = False
        elif mode == "processed":
            data = self._setup_processed()
            load_split_indices = True
        else:
            raise ValueError(f"Unknown data mode: {mode}. Use 'synthetic', 'h5ad', or 'processed'.")

        # Store metadata
        self.label_mapping = data.get("label_mapping", {})
        self.use_protein = data.get("use_protein", True)
        self.use_pseudotime = data.get("use_pseudotime", False)
        self.data_summary = data.get("data_summary", {})

        labels = data["labels"]
        if labels is None:
            labels = np.zeros(data["x_rna"].shape[0], dtype=np.int64)

        if load_split_indices:
            # Load pre-computed split indices
            split_path = os.path.join(self.config.data.processed_dir, "split_indices.json")
            if os.path.exists(split_path):
                with open(split_path, "r") as f:
                    sd = json.load(f)
                self.train_indices = np.array(sd["train"])
                self.val_indices = np.array(sd["val"])
                self.test_indices = np.array(sd["test"])
            else:
                raise FileNotFoundError(
                    f"split_indices.json not found in {self.config.data.processed_dir}. "
                    f"Re-run preprocessing to generate splits."
                )
        else:
            train_ratio = self.config.data.get("train_ratio", 0.7)
            val_ratio = self.config.data.get("val_ratio", 0.15)
            test_ratio = self.config.data.get("test_ratio", 0.15)
            train_idx, val_idx, test_idx = stratified_split(
                labels, train_ratio=train_ratio, val_ratio=val_ratio,
                test_ratio=test_ratio, seed=self.seed,
            )
            self.train_indices = train_idx
            self.val_indices = val_idx
            self.test_indices = test_idx

        # Subset adjacency for training
        if data.get("adjacency") is not None:
            adj = data["adjacency"]
            self.adjacency = torch.tensor(
                adj[self.train_indices][:, self.train_indices], dtype=torch.float32
            )

        # Create subset datasets
        def subset_data(indices):
            return {
                "x_rna": data["x_rna"][indices],
                "x_protein": data["x_protein"][indices],
                "labels": labels[indices],
                "pseudotime": data["pseudotime"][indices] if data.get("pseudotime") is not None else None,
                "adjacency": None,
                "label_mapping": self.label_mapping,
                "use_protein": self.use_protein,
                "use_pseudotime": self.use_pseudotime,
            }

        self.train_dataset = SingleCellMultiModalDataset(subset_data(self.train_indices), mode=mode)
        self.val_dataset = SingleCellMultiModalDataset(subset_data(self.val_indices), mode=mode)
        self.test_dataset = SingleCellMultiModalDataset(subset_data(self.test_indices), mode=mode)

    def _setup_synthetic(self):
        """Generate synthetic toy data."""
        cfg = self.config.data
        x_rna, x_protein, labels, pseudotime, adjacency = synthetic_data_generator(
            num_cells=cfg.num_cells,
            rna_dim=cfg.rna_dim,
            protein_dim=cfg.protein_dim,
            num_classes=cfg.num_classes,
            seed=self.seed,
        )
        num_classes = cfg.get("num_classes", 5)
        label_mapping = {str(i): f"class_{i}" for i in range(num_classes)}
        data_summary = {
            "n_cells": int(x_rna.shape[0]),
            "rna_dim": int(x_rna.shape[1]),
            "protein_dim": int(x_protein.shape[1]),
            "n_classes": num_classes,
            "use_protein": True,
            "use_pseudotime": True,
            "data_source": "synthetic",
        }
        return {
            "x_rna": x_rna,
            "x_protein": x_protein,
            "labels": labels,
            "pseudotime": pseudotime,
            "adjacency": adjacency,
            "label_mapping": label_mapping,
            "data_summary": data_summary,
            "use_protein": True,
            "use_pseudotime": True,
        }

    def _setup_h5ad(self):
        """Load data from an AnnData file."""
        return _load_h5ad_data(self.config.data)

    def _setup_processed(self):
        """Load preprocessed cached data."""
        data = _load_processed_data(self.config.data)
        # Merge data_summary if available
        sum_path = os.path.join(self.config.data.processed_dir, "data_summary.json")
        if os.path.exists(sum_path):
            with open(sum_path, "r") as f:
                data["data_summary"] = json.load(f)
        return data

    def train_dataloader(self):
        n_samples = len(self.train_dataset)
        drop = n_samples > self.batch_size
        return DataLoader(
            self.train_dataset,
            batch_size=min(self.batch_size, max(1, n_samples)),
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=drop,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def save_split_indices(self, filepath: str):
        save_split_indices(self.train_indices, self.val_indices, self.test_indices, filepath)

    @property
    def num_classes(self):
        labels = self.train_dataset.labels
        if labels is not None:
            return int(len(torch.unique(labels)))
        return self.config.data.get("num_classes", 1)

    @property
    def rna_dim(self):
        return self.train_dataset.x_rna.shape[1]

    @property
    def protein_dim(self):
        return self.train_dataset.x_protein.shape[1]
