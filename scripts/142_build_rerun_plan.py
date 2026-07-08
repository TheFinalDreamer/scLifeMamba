#!/usr/bin/env python3
"""142_build_rerun_plan.py — Determine which experiments need rerunning."""
import json
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.project_paths import LOCAL_RECOVERY_DIR

# Load data audit
data_inv = {}
data_json = LOCAL_RECOVERY_DIR / "local_data_inventory.json"
if data_json.exists():
    with open(data_json, 'r', encoding='utf-8') as f:
        data_inv = json.load(f)

data_status = data_inv.get("data_status", {})
results_exist = data_inv.get("results_exist", {})

# Environment status
NO_GPU = True
NO_MAMBA = True
FALLBACK_ONLY = True
NO_RAW_DATA = not data_status.get("pbmc_citeseq_raw", {}).get("found", False)
NO_SEQUENCES = not data_status.get("trajectory_sequences", {}).get("found", False)
NO_LABELS = not data_status.get("lifecycle_labels", {}).get("found", False)
LEGACY_RESULTS_EXIST = any(results_exist.values())

# Define rerun plan
RERUN_PLAN = [
    # P0 - Must recover or rerun
    {
        "task": "lifecycle_stage_labels",
        "priority": "P0",
        "current_status": "legacy results exist in code/output/",
        "required_data": "pbmc_citeseq_raw, pseudotime",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "30min (CPU)",
        "recommended_device": "CPU ok",
        "reason": "No raw .h5ad data. Existing labels can be loaded from legacy results.",
        "output_dir": "outputs/local_rerun/lifecycle_labels/",
    },
    {
        "task": "future_lifecycle_prediction",
        "priority": "P0",
        "current_status": "72 runs completed in code/output/ with metrics.json",
        "required_data": "trajectory_sequences, lifecycle_labels",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "~4h GPU (72 runs)",
        "recommended_device": "GPU (CUDA)",
        "reason": "Legacy results exist and are complete. Use existing metrics for paper.",
        "output_dir": "outputs/local_rerun/lifecycle_prediction/",
    },
    {
        "task": "future_pseudotime_regression",
        "priority": "P0",
        "current_status": "72 runs completed in code/output/",
        "required_data": "trajectory_sequences, pseudotime",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "~3h GPU (72 runs)",
        "recommended_device": "GPU (CUDA)",
        "reason": "No regression-specific metrics.json found with 'regression' keyword. Use results from existing metrics.",
        "output_dir": "outputs/local_rerun/pseudotime_regression/",
    },
    {
        "task": "dynamic_fusion_ablation",
        "priority": "P0",
        "current_status": "108 runs completed in code/output/",
        "required_data": "trajectory_sequences, lifecycle_labels",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "~6h GPU (108 runs)",
        "recommended_device": "GPU (CUDA)",
        "reason": "Legacy results complete. Use existing ablation metrics.",
        "output_dir": "outputs/local_rerun/ablation/",
    },
    # P1 - High priority
    {
        "task": "revised_trajectory_direction_labels",
        "priority": "P1",
        "current_status": "code exists (130/131), labels not built",
        "required_data": "trajectory_sequences, lifecycle_labels",
        "data_available": False,
        "code_available": True,
        "env_available": True,
        "can_rerun_now": False,
        "estimated_runtime": "10min (CPU, if labels available)",
        "recommended_device": "CPU ok",
        "reason": "Code complete. Cannot run without sequences/labels. Can generate design doc with label schemes.",
        "output_dir": "outputs/local_rerun/revised_direction/",
    },
    {
        "task": "protein_dominance_analysis",
        "priority": "P1",
        "current_status": "code exists (133), can use existing metrics",
        "required_data": "ablation metrics (available in code/output/)",
        "data_available": True,
        "code_available": True,
        "env_available": True,
        "can_rerun_now": True,
        "estimated_runtime": "5min (CPU)",
        "recommended_device": "CPU ok",
        "reason": "Can analyze based on existing ablation results (protein 88.99% vs RNA 32.22%).",
        "output_dir": "outputs/local_rerun/protein_dominance/",
    },
    {
        "task": "rna_to_protein_phenotype_generation",
        "priority": "P1",
        "current_status": "code skeleton exists (134), baseline results show mean_prediction strongest",
        "required_data": "trajectory_sequences",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "~3h GPU",
        "recommended_device": "GPU (CUDA)",
        "reason": "Cannot run without data. Existing baseline results already show mean_prediction is strongest.",
        "output_dir": "outputs/local_rerun/rna_to_protein/",
    },
    # P2
    {
        "task": "atac_guided_protein_compensation",
        "priority": "P2",
        "current_status": "code exists (135/136), no ATAC data",
        "required_data": "ATAC+RNA+Protein multiome",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "N/A",
        "recommended_device": "GPU (CUDA)",
        "reason": "No ATAC data on this machine. Must wait for data acquisition.",
        "output_dir": "outputs/local_rerun/atac_compensation/",
    },
    {
        "task": "second_dataset_validation",
        "priority": "P2",
        "current_status": "code exists (106), no external data",
        "required_data": "external CITE-seq dataset",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "~4h GPU",
        "recommended_device": "GPU (CUDA)",
        "reason": "No second dataset. Must download/prepare external data first.",
        "output_dir": "outputs/local_rerun/second_dataset/",
    },
    {
        "task": "perturbation_controls",
        "priority": "P2",
        "current_status": "code exists (107), not run",
        "required_data": "trajectory_sequences",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "~2h GPU",
        "recommended_device": "GPU (CUDA)",
        "reason": "Need data to run.",
        "output_dir": "outputs/local_rerun/perturbation/",
    },
    {
        "task": "branch_transition_analysis",
        "priority": "P2",
        "current_status": "code exists (108), not run",
        "required_data": "trajectory_sequences, branch labels",
        "data_available": False,
        "code_available": True,
        "env_available": False,
        "can_rerun_now": False,
        "estimated_runtime": "~2h GPU",
        "recommended_device": "GPU (CUDA)",
        "reason": "Need data to run.",
        "output_dir": "outputs/local_rerun/branch_transition/",
    },
]


def main():
    print("=== Building Rerun Plan ===")

    # Save JSON
    plan = {
        "environment": {
            "no_gpu": NO_GPU,
            "no_mamba_ssm": NO_MAMBA,
            "fallback_only": FALLBACK_ONLY,
            "no_raw_data": NO_RAW_DATA,
            "no_sequences": NO_SEQUENCES,
            "no_labels": NO_LABELS,
            "legacy_results_exist": LEGACY_RESULTS_EXIST,
        },
        "tasks": RERUN_PLAN,
    }

    json_path = LOCAL_RECOVERY_DIR / "rerun_plan.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")

    # Save CSV
    csv_path = LOCAL_RECOVERY_DIR / "rerun_plan.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(RERUN_PLAN[0].keys()))
        writer.writeheader()
        writer.writerows(RERUN_PLAN)
    print(f"CSV saved: {csv_path}")

    # Print summary
    print("\n=== Rerun Plan Summary ===")
    for task in RERUN_PLAN:
        flag = "[NOW]" if task["can_rerun_now"] else "[WAIT]"
        print(f"  {flag} [{task['priority']}] {task['task']}: {task['reason'][:80]}")

    can_rerun = [t for t in RERUN_PLAN if t["can_rerun_now"]]
    print(f"\nCan rerun NOW: {len(can_rerun)} tasks")
    for t in can_rerun:
        print(f"  - {t['task']}: {t['estimated_runtime']}")

    p0_blocked = [t for t in RERUN_PLAN if t["priority"] == "P0" and not t["can_rerun_now"]]
    print(f"P0 blocked: {len(p0_blocked)} tasks")
    for t in p0_blocked:
        print(f"  - {t['task']}: {t['reason'][:80]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
