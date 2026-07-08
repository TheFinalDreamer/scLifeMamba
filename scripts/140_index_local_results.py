#!/usr/bin/env python3
"""140_index_local_results.py — Scan all possible result directories and build inventory."""
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.project_paths import (
    PROJECT_ROOT, CODE_DIR, OUTPUTS_DIR, LEGACY_OUTPUT_DIR,
    LOCAL_RECOVERY_DIR, MANUSCRIPT_DIR, DOCS_DIR, get_recovery_dir
)

SEARCH_DIRS = [
    OUTPUTS_DIR,
    LEGACY_OUTPUT_DIR,
    CODE_DIR / "outputs",
    PROJECT_ROOT / "results",
    CODE_DIR / "results",
    PROJECT_ROOT / "docs",
    DOCS_DIR,
    MANUSCRIPT_DIR,
    PROJECT_ROOT / "phase8_scripts",
    PROJECT_ROOT / "phase8_fix",
    PROJECT_ROOT / "server_scripts",
    PROJECT_ROOT / "archive",
    PROJECT_ROOT / "技术文档" / "_archive",
]

KEY_FILES = [
    "run_status.json", "metrics.json", "config.json", "training_log.json",
    "summary.csv", "summary.json", "results.csv", "report.md",
    "*.png", "*.svg", "*.pdf", "*.tex", "*.bib"
]

EXPERIMENT_GROUPS = {
    "P0_lifecycle_labels": ["101", "lifecycle_label", "lifecycle_stage_label"],
    "P0_lifecycle_prediction": ["102", "lifecycle_prediction", "future_lifecycle", "main_classification"],
    "P0_pseudotime_regression": ["103", "pseudotime_regression", "future_pseudotime"],
    "P0_trajectory_direction": ["104", "trajectory_direction", "direction_prediction"],
    "P0_ablation": ["105", "ablation", "fusion_mode"],
    "P0_second_dataset": ["106", "second_dataset"],
    "P1_protein_dominance": ["protein_dominance", "133"],
    "P1_rna_to_protein": ["rna_to_protein", "phenotype_generation", "cross_modal", "134"],
    "P1_atac_compensation": ["atac", "135", "136"],
    "P1_perturbation": ["perturbation", "107"],
    "P1_branch": ["branch", "108"],
    "P1_revised_direction": ["revised_direction", "131", "132"],
    "old_highdim": ["highdim", "real_mamba", "79", "80", "81", "82", "83"],
    "old_lag_aware": ["lag_aware", "89", "90"],
    "old_phase5": ["phase5"],
    "old_phase6": ["phase6"],
    "old_phase8": ["phase8"],
    "old_baseline": ["baseline", "81", "90"],
    "figures": ["figure", "fig", ".png", ".svg"],
    "tables": ["table", ".csv", "summary"],
    "reports": ["report", "audit", ".md"],
    "manuscript": ["main.tex", "supplementary.tex", "references.bib"],
}


def classify_experiment(path_str: str) -> str:
    """Classify a path into an experiment group."""
    path_lower = path_str.lower()
    for group, keywords in EXPERIMENT_GROUPS.items():
        for kw in keywords:
            if kw.lower() in path_lower:
                return group
    return "unknown"


def is_server_legacy(path_str: str) -> bool:
    """Check if path indicates legacy server results."""
    indicators = ["phase5", "phase6", "phase8", "highdim", "server_scripts"]
    return any(ind in path_str.lower() for ind in indicators)


def is_current_mainline(path_str: str) -> bool:
    """Check if path is part of current mainline."""
    if is_server_legacy(path_str):
        return False
    mainline = ["P0_", "131", "132", "133", "134", "lifecycle", "pseudotime",
                "ablation", "protein_dominance", "revised_direction"]
    return any(m in path_str.lower() for m in mainline)


def index_results():
    """Scan all directories and build result inventory."""
    inventory = []
    seen = set()

    for search_dir in SEARCH_DIRS:
        if not Path(search_dir).exists():
            continue
        for file_path in Path(search_dir).rglob("*"):
            if not file_path.is_file():
                continue
            fpath_str = str(file_path)
            if fpath_str in seen:
                continue
            seen.add(fpath_str)

            suffix = file_path.suffix.lower()
            if suffix not in ['.json', '.csv', '.md', '.png', '.svg', '.pdf',
                              '.tex', '.bib', '.pt', '.pth', '.npy', '.npz',
                              '.yaml', '.yml', '.txt', '.log', '.html']:
                continue

            exp_group = classify_experiment(fpath_str)
            is_json = suffix == '.json'
            is_csv = suffix == '.csv'

            entry = {
                "file_path": fpath_str,
                "file_type": suffix,
                "experiment_group": exp_group,
                "task": "",
                "model": "",
                "seed": "",
                "horizon": "",
                "status": "found",
                "has_metrics": False,
                "has_config": False,
                "has_log": False,
                "is_complete": False,
                "is_server_legacy": is_server_legacy(fpath_str),
                "is_current_mainline": is_current_mainline(fpath_str),
                "can_be_used_in_main_text": False,
                "can_be_used_in_supplementary": False,
                "needs_rerun": False,
                "reason": "",
            }

            # Try to read JSON files for metadata
            if is_json and file_path.stat().st_size < 10_000_000:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        entry["has_metrics"] = any(k in data for k in ["metrics", "accuracy", "f1", "mse", "r2", "macro_f1"])
                        entry["has_config"] = any(k in data for k in ["config", "model", "architecture"])
                        entry["status"] = data.get("status", data.get("state", "unknown"))
                        entry["task"] = data.get("task", "")
                        entry["model"] = data.get("model", data.get("model_name", ""))
                        if "is_real_mamba" in data:
                            entry["is_real_mamba"] = data["is_real_mamba"]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            if suffix in ['.csv']:
                entry["has_metrics"] = True

            # Determine usability
            if entry["is_server_legacy"]:
                entry["can_be_used_in_supplementary"] = True
                entry["reason"] = "server legacy, supplementary only"
            elif entry["is_current_mainline"] and exp_group.startswith("P0"):
                if entry["has_metrics"]:
                    entry["can_be_used_in_main_text"] = True
                    entry["reason"] = "P0 mainline with metrics"
                else:
                    entry["can_be_used_in_supplementary"] = True
                    entry["needs_rerun"] = True
                    entry["reason"] = "P0 but missing metrics"
            elif exp_group.startswith("P1"):
                entry["can_be_used_in_supplementary"] = True
                entry["reason"] = "P1 experiment"

            inventory.append(entry)

    return inventory


def main():
    print("=== Indexing Local Results ===")
    inventory = index_results()
    print(f"Total indexed files: {len(inventory)}")

    # Save CSV
    csv_path = get_recovery_dir("") / "local_result_inventory.csv"
    if inventory:
        fieldnames = list(inventory[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inventory)
        print(f"CSV saved: {csv_path}")

    # Save JSON
    json_path = get_recovery_dir("") / "local_result_inventory.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON saved: {json_path}")

    # Summary
    groups = {}
    for item in inventory:
        g = item["experiment_group"]
        groups[g] = groups.get(g, 0) + 1

    print("\n=== By Experiment Group ===")
    for g, count in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g}: {count}")

    server_legacy = sum(1 for i in inventory if i["is_server_legacy"])
    mainline = sum(1 for i in inventory if i["is_current_mainline"])
    main_text = sum(1 for i in inventory if i["can_be_used_in_main_text"])
    print(f"\nServer legacy: {server_legacy}")
    print(f"Current mainline: {mainline}")
    print(f"Can be used in main text: {main_text}")

    print("\nDone.")


if __name__ == "__main__":
    main()
