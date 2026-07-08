#!/usr/bin/env python3
"""141_audit_local_data_availability.py — Scan data directories and assess completeness."""
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.project_paths import (
    PROJECT_ROOT, CODE_DIR, DATA_DIR, LEGACY_OUTPUT_DIR, OUTPUTS_DIR, get_recovery_dir
)

DATA_EXTENSIONS = ['.h5ad', '.h5', '.loom', '.mtx', '.csv', '.tsv',
                   '.parquet', '.npy', '.npz', '.pt', '.pkl']

REQUIRED_DATA = {
    "pbmc_citeseq_raw": {
        "description": "PBMC CITE-seq raw .h5ad file",
        "patterns": ["pbmc_citeseq", "citeseq"],
        "extensions": [".h5ad", ".h5"],
        "essential": True,
    },
    "rna_protein_data": {
        "description": "RNA+Protein paired expression data",
        "patterns": ["rna", "protein", "citeseq"],
        "extensions": [".h5ad", ".npy", ".npz"],
        "essential": True,
    },
    "trajectory_sequences": {
        "description": "Pre-built trajectory window sequences",
        "patterns": ["sequence", "trajectory", "window", "train_sequences", "test_sequences"],
        "extensions": [".pt", ".npy"],
        "essential": True,
    },
    "lifecycle_labels": {
        "description": "Lifecycle stage classification labels",
        "patterns": ["lifecycle", "label", "stage"],
        "extensions": [".csv", ".npy"],
        "essential": True,
    },
    "pseudotime": {
        "description": "Pseudotime values per cell",
        "patterns": ["pseudotime", "dpt"],
        "extensions": [".csv", ".npy", ".h5ad"],
        "essential": True,
    },
    "atac_data": {
        "description": "ATAC-seq or multiome data",
        "patterns": ["atac", "multiome", "chromatin"],
        "extensions": [".h5ad", ".h5", ".mtx", ".tsv"],
        "essential": False,
    },
    "second_dataset": {
        "description": "External validation dataset",
        "patterns": ["bmmc", "asap", "external", "second"],
        "extensions": [".h5ad", ".h5"],
        "essential": False,
    },
    "pretrained_models": {
        "description": "Pre-trained model checkpoints",
        "patterns": ["best_model", "last_model", "checkpoint"],
        "extensions": [".pth", ".pt"],
        "essential": False,
    },
}


def scan_data_files():
    """Scan all data directories for known file types."""
    search_dirs = [
        DATA_DIR,
        CODE_DIR / "data",
        PROJECT_ROOT / "datasets",
        CODE_DIR / "datasets",
        LEGACY_OUTPUT_DIR,
        OUTPUTS_DIR,
    ]

    found_files = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in DATA_EXTENSIONS:
            for f in search_dir.rglob(f"*{ext}"):
                found_files.append(str(f))

    return found_files


def check_data_availability(found_files: list) -> dict:
    """Check which required data items are available."""
    results = {}
    for key, spec in REQUIRED_DATA.items():
        matches = []
        for f in found_files:
            f_lower = f.lower()
            if any(p in f_lower for p in spec["patterns"]):
                if any(f_lower.endswith(ext) for ext in spec["extensions"]):
                    matches.append(f)

        results[key] = {
            "description": spec["description"],
            "essential": spec["essential"],
            "found": len(matches) > 0,
            "count": len(matches),
            "files": matches[:5],  # first 5
            "status": "available" if matches else ("missing_essential" if spec["essential"] else "missing_optional"),
        }
    return results


def check_p0_inputs(data_status: dict) -> dict:
    """Check if P0 experiment inputs are available."""
    p0_tasks = {
        "lifecycle_label_construction": {
            "requires": ["pbmc_citeseq_raw", "pseudotime"],
            "status": "unknown",
        },
        "lifecycle_prediction": {
            "requires": ["trajectory_sequences", "lifecycle_labels"],
            "status": "unknown",
        },
        "pseudotime_regression": {
            "requires": ["trajectory_sequences", "pseudotime"],
            "status": "unknown",
        },
        "trajectory_direction": {
            "requires": ["trajectory_sequences", "lifecycle_labels"],
            "status": "unknown",
        },
        "ablation": {
            "requires": ["trajectory_sequences", "lifecycle_labels"],
            "status": "unknown",
        },
        "protein_dominance": {
            "requires": ["rna_protein_data"],
            "status": "unknown",
        },
        "rna_to_protein": {
            "requires": ["rna_protein_data", "trajectory_sequences"],
            "status": "unknown",
        },
        "atac_compensation": {
            "requires": ["atac_data", "rna_protein_data"],
            "status": "unknown",
        },
        "second_dataset": {
            "requires": ["second_dataset"],
            "status": "unknown",
        },
    }

    for task, info in p0_tasks.items():
        all_available = all(data_status.get(req, {}).get("found", False) for req in info["requires"])
        any_missing = any(data_status.get(req, {}).get("status") == "missing_essential" for req in info["requires"])
        if all_available:
            info["status"] = "ready"
        elif any_missing:
            info["status"] = "blocked_missing_essential"
        else:
            info["status"] = "blocked_missing_optional"

    return p0_tasks


def check_results_exist() -> dict:
    """Check if results already exist in legacy output dir."""
    result_indicators = {
        "lifecycle_prediction_results": False,
        "pseudotime_regression_results": False,
        "ablation_results": False,
        "direction_results": False,
    }

    if LEGACY_OUTPUT_DIR.exists():
        for d in LEGACY_OUTPUT_DIR.rglob("metrics.json"):
            p = str(d)
            if "main_classification" in p:
                result_indicators["lifecycle_prediction_results"] = True
            if "ablation" in p:
                result_indicators["ablation_results"] = True

        for d in LEGACY_OUTPUT_DIR.glob("results/*/metrics.json"):
            p = str(d)
            if "regression" in p.lower():
                result_indicators["pseudotime_regression_results"] = True

    return result_indicators


def main():
    print("=== Data Availability Audit ===")
    found_files = scan_data_files()
    print(f"Total data files found: {len(found_files)}")

    data_status = check_data_availability(found_files)
    p0_inputs = check_p0_inputs(data_status)
    results_exist = check_results_exist()

    # Save JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_status": data_status,
        "p0_inputs": p0_inputs,
        "results_exist": results_exist,
        "total_data_files": len(found_files),
        "data_files": found_files,
    }

    out_dir = get_recovery_dir("")
    json_path = out_dir / "local_data_inventory.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON saved: {json_path}")

    # CSV summary
    csv_path = out_dir / "local_data_inventory.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["data_item", "essential", "found", "count", "status"])
        for key, info in data_status.items():
            writer.writerow([key, info["essential"], info["found"], info["count"], info["status"]])

    print("\n=== Data Status ===")
    for key, info in data_status.items():
        flag = "[OK]" if info["found"] else ("[MISS]" if info["essential"] else "[OPT]")
        print(f"  {flag} {key}: {info['status']} ({info['count']} files)")

    print("\n=== P0 Input Readiness ===")
    for task, info in p0_inputs.items():
        flag = "[OK]" if info["status"] == "ready" else "[MISS]"
        print(f"  {flag} {task}: {info['status']}")

    print("\n=== Legacy Results Exist ===")
    for key, val in results_exist.items():
        flag = "[OK]" if val else "[NO]"
        print(f"  {flag} {key}")

    print("\nDone.")


if __name__ == "__main__":
    main()
