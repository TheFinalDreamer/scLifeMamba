"""Utility functions for I/O, logging, and hash verification."""
import os
import sys
import json
import yaml
import hashlib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# Paths — auto-detect project root
# ---------------------------------------------------------------------------
def _find_root() -> Path:
    """Find project root by searching for marker directories."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "src").is_dir() and (current / "scripts").is_dir():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = _find_root()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = Path(os.getenv("SCLIFEMAMBA_OUTPUT", PROJECT_ROOT / "outputs"))
LOGS_DIR = OUTPUTS / "logs"
REPORTS_DIR = OUTPUTS / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    ensure_dirs(LOGS_DIR)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        log_file = Path(log_file)
        if not log_file.is_absolute():
            log_file = LOGS_DIR / log_file
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# File integrity
# ---------------------------------------------------------------------------
def sha256_hex(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------
def disk_free_gb(path: Path = Path("/data/sc")) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def disk_total_gb(path: Path = Path("/data/sc")) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.total / (1024 ** 3)
