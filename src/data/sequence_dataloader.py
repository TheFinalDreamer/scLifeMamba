"""
Sequence-aware data pipeline for Mamba-LSTM experiments.

Loads leakage_safe_v1 preprocessed data and produces sequence tensors
with separate RNA (1000 HVG) and Protein (228 ADT) channels.

Usage:
    loader = SequenceDataLoader(data_dir, horizon=1, batch_size=64)
    for batch in loader.train():
        x_rna = batch['x_rna']      # (B, 32, 1000)
        x_protein = batch['x_protein']  # (B, 32, 228)
        labels = batch['label']     # (B,)

Supports:
    - Original pseudotime-ordered sequences
    - Random shuffled sequences (for Exp 3)
    - No-order (bag-of-cells) sequences (for Exp 3)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LeakageSafeSequenceDataset(Dataset):
    """Dataset loading RNA+Protein sequences from leakage_safe_v1 preprocessed data.

    Each item is a sequence of L=32 cells, each with RNA (1000 HVG) and Protein (228 ADT)
    features, plus a target lifecycle label.

    Args:
        manifest_df: Filtered pandas DataFrame with sequence metadata
        rna_features: np.memmap or np.ndarray (n_cells, 1000)
        protein_features: np.memmap or np.ndarray (n_cells, 228)
        shuffle_order: 'original' | 'random' | 'no_order'
        seed: Random seed for shuffling
    """

    def __init__(self, manifest_df, rna_features, protein_features,
                 shuffle_order='original', seed=42):
        self.manifest = manifest_df.reset_index(drop=True)
        self.rna = rna_features
        self.protein = protein_features
        self.shuffle_order = shuffle_order
        self.rng = np.random.RandomState(seed)

        # Count classes
        self.n_classes = self.manifest['target_label'].nunique()

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        # Parse context cell indices — supports both parquet (cell UIDs) and CSV (JSON indices)
        if 'context_cell_uids' in row.index:
            # Parquet format: cell_041868|cell_029079|... → extract numeric indices
            uids_str = row['context_cell_uids']
            if isinstance(uids_str, str):
                indices = np.array([int(uid.split('_')[1]) for uid in uids_str.split('|')], dtype=np.int64)
            else:
                indices = np.array([int(uid.split('_')[1]) for uid in uids_str], dtype=np.int64)
        elif 'context_indices' in row.index:
            # CSV format: JSON-encoded indices list
            indices_str = row['context_indices']
            if isinstance(indices_str, str):
                indices = np.array(json.loads(indices_str), dtype=np.int64)
            else:
                indices = np.array(indices_str, dtype=np.int64)
        else:
            raise KeyError("Manifest must have 'context_cell_uids' or 'context_indices' column")

        seq_len = len(indices)  # should be 32

        # Shuffle if needed
        if self.shuffle_order == 'random':
            indices = indices.copy()
            self.rng.shuffle(indices)
        elif self.shuffle_order == 'no_order':
            # Sort by feature magnitude to destroy temporal structure
            # (random permutation seeded differently per sequence)
            rng_seq = np.random.RandomState(idx + 999983)
            indices = indices.copy()
            rng_seq.shuffle(indices)

        # Extract features
        x_rna = torch.as_tensor(self.rna[indices].copy(), dtype=torch.float32)
        x_protein = torch.as_tensor(self.protein[indices].copy(), dtype=torch.float32)
        label = torch.tensor(int(row['target_label']), dtype=torch.long)

        # Get sequence ID (parquet: reference_sequence_id, csv: sequence_id)
        seq_id = row.get('reference_sequence_id', row.get('sequence_id', str(idx)))
        return {
            'x_rna': x_rna,           # (seq_len, rna_dim)
            'x_protein': x_protein,   # (seq_len, protein_dim)
            'label': label,           # scalar
            'sequence_id': str(seq_id),
        }


class SequenceDataLoader:
    """Data loader factory for Mamba-LSTM experiments.

    Loads the leakage_safe_v1 data and creates train/val/test DataLoaders
    for a specified horizon and sequence ordering.
    """

    def __init__(
        self,
        data_dir: str,
        horizon: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle_order: str = 'original',
        seed: int = 42,
        max_sequences: int = None,
    ):
        """
        Args:
            data_dir: Path to leakage_safe_v1 processed data
            horizon: Prediction horizon (1, 4, 8, or 16)
            batch_size: Batch size
            num_workers: DataLoader workers (0 = main process)
            shuffle_order: 'original', 'random', or 'no_order'
            seed: Random seed
            max_sequences: Limit number of sequences (for debugging)
        """
        self.data_dir = data_dir
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_order = shuffle_order
        self.seed = seed

        # Load features as memory-mapped arrays
        rna_path = os.path.join(data_dir, 'rna_hvg_all.npy')
        protein_path = os.path.join(data_dir, 'protein_norm_all.npy')

        if not os.path.exists(rna_path):
            raise FileNotFoundError(f"RNA features not found: {rna_path}")
        if not os.path.exists(protein_path):
            raise FileNotFoundError(f"Protein features not found: {protein_path}")

        self.rna_features = np.load(rna_path, mmap_mode='r')
        self.protein_features = np.load(protein_path, mmap_mode='r')

        print(f"Loaded features: RNA {self.rna_features.shape}, Protein {self.protein_features.shape}")

        # Load sequence manifest
        self._load_manifest(horizon, max_sequences)

        # Create datasets
        self._create_datasets()

    def _load_manifest(self, horizon, max_sequences):
        """Load and filter the sequence manifest for the given horizon. Uses parquet for speed."""
        import pandas as pd

        # Prefer fast parquet format (0.5s vs 60s for CSV)
        pq_path = os.path.join(self.data_dir, 'reference_sequence_manifest.parquet')
        csv_path = os.path.join(self.data_dir, 'sequence_manifest.csv')

        if os.path.exists(pq_path):
            df = pd.read_parquet(pq_path)
            self.manifest = df[df['horizon'] == horizon].copy()
            print(f"Loaded {len(self.manifest)} sequences for horizon={horizon} (parquet, fast)")
        elif os.path.exists(csv_path):
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=50000):
                chunk = chunk[chunk['horizon'] == horizon]
                if len(chunk) > 0:
                    chunks.append(chunk)
            if not chunks:
                raise ValueError(f"No sequences found for horizon={horizon}")
            self.manifest = pd.concat(chunks, ignore_index=True)
            print(f"Loaded {len(self.manifest)} sequences for horizon={horizon} (csv, slow)")
        else:
            raise FileNotFoundError(f"No manifest found in {self.data_dir}")

        if max_sequences is not None and max_sequences < len(self.manifest):
            self.manifest = self.manifest.sample(n=max_sequences, random_state=self.seed)
            print(f"  Subsampled to {max_sequences} sequences")

    def _create_datasets(self):
        """Create train/val/test datasets."""
        train_df = self.manifest[self.manifest['split'] == 'train'].copy()
        val_df = self.manifest[self.manifest['split'] == 'val'].copy()
        test_df = self.manifest[self.manifest['split'] == 'test'].copy()

        print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        self.train_dataset = LeakageSafeSequenceDataset(
            train_df, self.rna_features, self.protein_features,
            shuffle_order=self.shuffle_order, seed=self.seed,
        )
        self.val_dataset = LeakageSafeSequenceDataset(
            val_df, self.rna_features, self.protein_features,
            shuffle_order=self.shuffle_order, seed=self.seed,
        )
        self.test_dataset = LeakageSafeSequenceDataset(
            test_df, self.rna_features, self.protein_features,
            shuffle_order=self.shuffle_order, seed=self.seed,
        )

        self.n_classes = self.train_dataset.n_classes
        # Feature dimensions
        sample = self.train_dataset[0]
        self.rna_dim = sample['x_rna'].shape[-1]
        self.protein_dim = sample['x_protein'].shape[-1]
        self.seq_len = sample['x_rna'].shape[0]

        print(f"Feature dims: rna={self.rna_dim}, protein={self.protein_dim}, seq_len={self.seq_len}")
        print(f"Number of classes: {self.n_classes}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    @property
    def train_size(self):
        return len(self.train_dataset)

    @property
    def val_size(self):
        return len(self.val_dataset)

    @property
    def test_size(self):
        return len(self.test_dataset)
