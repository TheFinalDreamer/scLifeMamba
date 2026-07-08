"""Timestamp and run directory utilities."""

from datetime import datetime
import os


def get_timestamp():
    """Return current timestamp string YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(root_dir: str, experiment_name: str) -> dict:
    """Create timestamped run directories for an experiment.

    Returns a dict with paths: run_dir, checkpoint_dir, figure_dir, log_dir, result_dir.
    """
    ts = get_timestamp()
    run_name = f"{ts}_{experiment_name}"

    run_dir = os.path.join(root_dir, "results", run_name)
    checkpoint_dir = os.path.join(root_dir, "checkpoints", run_name)
    figure_dir = os.path.join(root_dir, "figures", run_name)
    log_dir = os.path.join(root_dir, "logs", run_name)

    for d in [run_dir, checkpoint_dir, figure_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "figure_dir": figure_dir,
        "log_dir": log_dir,
        "run_name": run_name,
        "timestamp": ts,
    }
