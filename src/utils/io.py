"""I/O utilities for saving/loading JSON, CSV, numpy arrays, and checkpoints."""

import json
import os
import csv
import numpy as np
import torch
import yaml


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, filepath: str):
    """Save a dict as JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_json(filepath: str) -> dict:
    """Load a JSON file into a dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(data: list[dict], filepath: str, fieldnames: list = None):
    """Save a list of dicts as CSV."""
    ensure_dir(os.path.dirname(filepath))
    if not data:
        return
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def save_config_used(config: dict, filepath: str):
    """Save the config used for an experiment as YAML."""
    ensure_dir(os.path.dirname(filepath))
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def save_npy(data: np.ndarray, filepath: str):
    """Save numpy array to .npy file."""
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, data)


def save_model_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath: str):
    """Save a full training checkpoint."""
    ensure_dir(os.path.dirname(filepath))
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model, optimizer, scheduler, filepath: str, device: str = "cpu"):
    """Load a full training checkpoint."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
