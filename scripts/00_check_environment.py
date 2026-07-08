#!/usr/bin/env python3
"""00_check_environment.py — Comprehensive server environment check.

Generates: docs/ENVIRONMENT_CHECK.md
Run: python code/scripts/00_check_environment.py
Safe for repeated runs.
"""
import sys
import os
import platform
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(os.environ.get("SCLIFEMAMBA_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from utils.utils_io import (
    ensure_dirs, timestamp, now_iso, save_json, setup_logger,
    DOCS_DIR, LOGS_DIR, REPORTS_DIR, PROJECT_ROOT,
)

logger = setup_logger("env_check", LOGS_DIR / "environment_check.log")


def run(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"[ERROR] {e}"


def check_python() -> dict:
    import numpy, pandas, scipy, sklearn, torch, anndata, scanpy
    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
    }
    for name, mod in [("numpy", numpy), ("pandas", pandas), ("scipy", scipy),
                       ("scikit-learn", sklearn), ("torch", torch),
                       ("anndata", anndata), ("scanpy", scanpy)]:
        try:
            info[name] = mod.__version__
        except Exception:
            info[name] = "N/A"

    info["torch_cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    # Additional packages
    for pkg in ["mudata", "muon", "pooch", "gdown", "GEOparse", "pyarrow",
                 "igraph", "leidenalg", "pytorch_lightning", "matplotlib", "seaborn", "h5py"]:
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            info[pkg] = "MISSING"

    return info


def check_system() -> dict:
    info = {}
    info["hostname"] = platform.node()
    info["os"] = f"{platform.system()} {platform.release()}"
    info["cpu_count"] = os.cpu_count()

    # Disk space on /data
    usage = shutil.disk_usage("/data/sc")
    info["disk_total_gb"] = round(usage.total / (1024 ** 3), 1)
    info["disk_used_gb"] = round(usage.used / (1024 ** 3), 1)
    info["disk_free_gb"] = round(usage.free / (1024 ** 3), 1)

    # Memory
    mem = run("free -h | grep Mem:")
    info["memory"] = mem

    # GPU
    nvidia = run("nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader")
    info["nvidia_smi"] = nvidia

    # Conda env
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "N/A")
    info["conda_env"] = conda_env

    return info


def main():
    logger.info("=" * 60)
    logger.info("Phase 3 — Environment Check")
    logger.info(f"Time: {now_iso()}")
    logger.info(f"Project: {PROJECT_ROOT}")
    logger.info("=" * 60)

    results = {
        "timestamp": now_iso(),
        "project_root": str(PROJECT_ROOT),
    }

    logger.info("Checking system resources...")
    results["system"] = check_system()
    for k, v in results["system"].items():
        logger.info(f"  {k}: {v}")

    logger.info("Checking Python packages...")
    results["python"] = check_python()
    for k, v in results["python"].items():
        logger.info(f"  {k}: {v}")

    # Project structure check
    required_dirs = [
        "code/src/data", "code/src/models", "code/src/training", "code/src/utils",
        "code/scripts", "code/configs/dataset", "code/configs/experiment",
        "docs", "outputs/logs", "outputs/reports", "outputs/figures",
        "outputs/checkpoints", "outputs/cache", "data/raw", "data/processed",
    ]
    existing = []
    missing = []
    for d in required_dirs:
        dp = PROJECT_ROOT / d
        if dp.exists():
            existing.append(d)
        else:
            missing.append(d)
            dp.mkdir(parents=True, exist_ok=True)
            logger.warning(f"  Created missing dir: {d}")

    results["project_dirs"] = {"existing": existing, "created": missing}

    # Save JSON summary
    json_path = DOCS_DIR / "environment_check.json"
    save_json(results, json_path)
    logger.info(f"JSON saved: {json_path}")

    # Generate Markdown report
    md = generate_markdown(results)
    md_path = DOCS_DIR / "ENVIRONMENT_CHECK.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"Report saved: {md_path}")

    # Summary
    health = "PASS"
    issues = []
    if not results["python"]["torch_cuda_available"]:
        issues.append("CUDA not available")
        health = "FAIL"
    for pkg, ver in results["python"].items():
        if ver == "MISSING":
            issues.append(f"Missing package: {pkg}")
            health = "WARN"

    logger.info(f"Overall health: {health}")
    if issues:
        for i in issues:
            logger.warning(f"  Issue: {i}")

    return 0 if health == "PASS" else 1


def generate_markdown(data: dict) -> str:
    s = data["system"]
    p = data["python"]
    lines = [
        f"# Environment Check Report",
        f"",
        f"> Generated: {data['timestamp']}",
        f"> Project: {data['project_root']}",
        f"",
        f"## System",
        f"",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Hostname | {s['hostname']} |",
        f"| OS | {s['os']} |",
        f"| CPU Cores | {s['cpu_count']} |",
        f"| Disk Total | {s['disk_total_gb']} GB |",
        f"| Disk Free | {s['disk_free_gb']} GB |",
        f"| Conda Env | {s['conda_env']} |",
        f"",
        f"### GPU",
        f"```",
        s.get("nvidia_smi", "N/A"),
        f"```",
        f"",
        f"### Memory",
        f"```",
        s.get("memory", "N/A"),
        f"```",
        f"",
        f"## Python Packages",
        f"",
        f"| Package | Version |",
        f"|---------|---------|",
    ]
    for name in ["python_version", "numpy", "pandas", "scipy", "scikit-learn",
                  "torch", "cuda_version", "torch_cuda_available",
                  "anndata", "scanpy", "mudata", "muon", "pooch", "gdown",
                  "GEOparse", "pyarrow", "igraph", "leidenalg",
                  "pytorch_lightning", "matplotlib", "seaborn", "h5py"]:
        val = p.get(name, "N/A")
        lines.append(f"| {name} | {val} |")

    lines += [
        f"",
        f"### GPU Details",
    ]
    for i, gpu in enumerate(p.get("gpu_names", [])):
        lines.append(f"- GPU {i}: {gpu}")

    lines += [
        f"",
        f"## Project Structure",
        f"",
        f"| Directory | Status |",
        f"|-----------|--------|",
    ]
    for d in data["project_dirs"]["existing"]:
        lines.append(f"| {d} | OK |")
    for d in data["project_dirs"]["created"]:
        lines.append(f"| {d} | CREATED |")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
