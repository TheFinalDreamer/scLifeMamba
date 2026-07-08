"""
project_paths.py — Unified project path resolution for scLifeMamba.

Priority:
  1. Environment variable SCLIFEMAMBA_ROOT
  2. Auto-detect by searching upward for 'code/', '技术文档/', 'manuscript/'
  3. Fallback to current working directory

Usage:
    from src.utils.project_paths import (
        PROJECT_ROOT, CODE_DIR, DATA_DIR, OUTPUTS_DIR, LEGACY_OUTPUT_DIR,
        MANUSCRIPT_DIR, DOCS_DIR, get_output_dir, get_rerun_dir
    )
"""
import os
import sys
from pathlib import Path
from datetime import datetime


def _find_project_root() -> Path:
    """Find project root by priority: env var > auto-detect > cwd."""
    # 1. Environment variable
    env_root = os.environ.get("SCLIFEMAMBA_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p.resolve()

    # 2. Auto-detect: search upward from this file
    current = Path(__file__).resolve().parent  # src/utils/
    for _ in range(10):
        # Check for marker directories
        if (current / "code").is_dir() and (current / "技术文档").is_dir() and (current / "manuscript").is_dir():
            return current
        # Also check if we're inside code/ (then parent is root)
        if current.name == "code" and (current.parent / "技术文档").is_dir() and (current.parent / "manuscript").is_dir():
            return current.parent
        current = current.parent

    # 3. Fallback: current working directory
    cwd = Path.cwd()
    if (cwd / "code").is_dir():
        return cwd
    return cwd


PROJECT_ROOT = _find_project_root()
CODE_DIR = PROJECT_ROOT / "code"
SRC_DIR = CODE_DIR / "src"
SCRIPTS_DIR = CODE_DIR / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LEGACY_OUTPUT_DIR = CODE_DIR / "output"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript" / "bioinformatics"
DOCS_DIR = PROJECT_ROOT / "技术文档" / "current"
ARCHIVE_DIR = PROJECT_ROOT / "技术文档" / "_archive"
LOCAL_RECOVERY_DIR = OUTPUTS_DIR / "local_recovery_audit"
LOCAL_RERUN_DIR = OUTPUTS_DIR / "local_rerun"


def get_output_dir(task_name: str, base_dir: Path = None) -> Path:
    """Create and return a timestamped output directory for a task."""
    if base_dir is None:
        base_dir = LOCAL_RERUN_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{timestamp}_{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_rerun_dir(task_name: str) -> Path:
    """Create and return a timestamped rerun directory."""
    return get_output_dir(task_name, base_dir=LOCAL_RERUN_DIR)


def get_recovery_dir(task_name: str) -> Path:
    """Create and return a directory under local_recovery_audit."""
    out_dir = LOCAL_RECOVERY_DIR / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# Ensure output directories exist
LOCAL_RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_RERUN_DIR.mkdir(parents=True, exist_ok=True)


# Print resolution info on import (for debugging)
def print_paths():
    """Print resolved paths for debugging."""
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"CODE_DIR:        {CODE_DIR}")
    print(f"DATA_DIR:        {DATA_DIR}")
    print(f"OUTPUTS_DIR:     {OUTPUTS_DIR}")
    print(f"LEGACY_OUTPUT:   {LEGACY_OUTPUT_DIR}")
    print(f"MANUSCRIPT_DIR:  {MANUSCRIPT_DIR}")
    print(f"DOCS_DIR:        {DOCS_DIR}")
    print(f"LOCAL_RECOVERY:  {LOCAL_RECOVERY_DIR}")
    print(f"LOCAL_RERUN:     {LOCAL_RERUN_DIR}")
    print(f"SRC_DIR:         {SRC_DIR}")


# Add code/ to sys.path if not already present
_code_str = str(CODE_DIR)
if _code_str not in sys.path:
    sys.path.insert(0, _code_str)


if __name__ == "__main__":
    print_paths()
