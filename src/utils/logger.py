"""Experiment logging utilities supporting JSONL file and console output."""

import json
import logging
import sys
import os


def setup_logger(name: str = "scLifeMamba", level: int = logging.INFO) -> logging.Logger:
    """Create a console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


class ExperimentLogger:
    """Logger that writes JSONL training logs and also prints to console."""

    def __init__(self, log_dir: str, experiment_name: str = "scLifeMamba"):
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, "training_log.jsonl")
        self.console = setup_logger(experiment_name)

    def log(self, record: dict):
        """Write a record to JSONL and console."""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self.console.info(self._format(record))

    def log_epoch(self, record: dict):
        """Convenience method for logging epoch results."""
        self.log(record)

    @staticmethod
    def _format(record: dict) -> str:
        parts = []
        for k, v in record.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        return " | ".join(parts)

    def save_metrics(self, metrics: dict, filepath: str):
        """Save final metrics as pretty JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str, ensure_ascii=False)
