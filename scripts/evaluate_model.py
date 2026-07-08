#!/usr/bin/env python
"""Standalone model evaluation script.

Usage:
    python code/scripts/evaluate_model.py --checkpoint path/to/best_model.pth --config path/to/config.yaml
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import build_config
from src.utils.seed import set_seed
from src.utils.io import save_json
from src.data.datamodule import MultiModalDataModule
from src.models.scLifeMamba import scLifeMamba
from src.training.trainer import Trainer
from src.evaluation.trajectory_metrics import compute_trajectory_metrics
from src.evaluation.efficiency import compute_efficiency_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--output", type=str, default=None, help="Output directory for evaluation results")
    args = parser.parse_args()

    config = build_config(args.config)
    set_seed(config.project.seed)

    # Data
    datamodule = MultiModalDataModule(config)
    datamodule.setup()

    # Model
    model = scLifeMamba(
        rna_dim=datamodule.rna_dim,
        protein_dim=datamodule.protein_dim,
        num_classes=datamodule.num_classes,
        rna_hidden_dims=config.model.get("rna_hidden_dims", [512, 256]),
        protein_hidden_dims=config.model.get("protein_hidden_dims", [64, 64]),
        hidden_dim=config.model.get("hidden_dim", 128),
        embedding_dim=config.model.get("embedding_dim", 64),
        dropout=config.model.get("dropout", 0.2),
        use_mamba=config.model.get("use_mamba", True),
        use_lstm=config.model.get("use_lstm", True),
        use_dynamic_fusion=config.model.get("use_dynamic_fusion", True),
    )

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create a minimal trainer for prediction
    run_dirs = {
        "run_dir": args.output or ".",
        "checkpoint_dir": args.output or ".",
        "figure_dir": args.output or ".",
        "log_dir": args.output or ".",
    }
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Dummy config wrapper for Trainer
    trainer = Trainer(model, datamodule, config, run_dirs)

    # Test
    test_metrics = trainer.test()
    print(f"Test accuracy: {test_metrics.get('accuracy', 'N/A')}")

    # Generate predictions
    predictions = trainer.predict(datamodule.test_dataloader())
    test_labels = datamodule.test_dataset.labels.numpy()

    # Trajectory metrics
    traj_metrics = compute_trajectory_metrics(
        predictions["embedding"], test_labels,
        datamodule.test_dataset.pseudotime.numpy() if datamodule.test_dataset.pseudotime is not None else None,
    )

    # Efficiency
    sample_batch = {
        "x_rna": torch.randn(64, datamodule.rna_dim).to(device),
        "x_protein": torch.randn(64, datamodule.protein_dim).to(device),
    }
    eff_metrics = compute_efficiency_metrics(model, sample_batch, str(device))

    all_metrics = {**test_metrics, **traj_metrics, **eff_metrics}
    output_path = os.path.join(args.output, "evaluation_metrics.json") if args.output else "evaluation_metrics.json"
    save_json(all_metrics, output_path)
    print(f"Evaluation complete. Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
