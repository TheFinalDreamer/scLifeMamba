# FINAL_GITHUB_RELEASE_REPORT

Date: 2026-07-19
Repository: `GITHUB_RELEASE/scLifeMamba`

## 1. GitHub是否与论文一致？

Yes. The release repository now contains the code paths required by the manuscript Code Availability statement and README:

- `scripts/run_mamba_final_experiments.py`
- `scripts/run_final_audit_experiments.py`
- `scripts/81_compute_baselines.py`
- `src/models/torch_selective_ssm.py`
- `src/data/`
- `src/dataset/`
- `configs/`
- `docs/`

README training commands now use 10 epochs, matching the manuscript and supplementary material.

## 2. 代码是否可复现？

Yes, subject to the documented data requirement. Raw PBMC CITE-seq data are not redistributed. The README documents the expected leakage-safe processed data directory and required files. Scripts use repository-relative defaults and accept `--data_dir` overrides.

Validation completed:

- Python syntax check for final scripts and core backend files
- Model smoke test for sequence input
- Backend check returning `torch_selective_ssm`
- README command path mapping

## 3. 是否存在缺失文件？

No required release file is missing after this pass.

Confirmed present:

- `README.md`
- `requirements.txt`
- `environment.yml`
- `scripts/run_mamba_final_experiments.py`
- `scripts/run_final_audit_experiments.py`
- `scripts/81_compute_baselines.py`
- `src/models/torch_selective_ssm.py`
- `src/models/mamba_block.py`
- `src/models/scLifeMamba.py`
- `src/data/sequence_dataloader.py`
- `src/dataset/sequence_dataloader.py`

## 4. 是否可以解除BLOCKED？

For the local release repository: yes. The previous local block was caused by missing release files and README epoch inconsistency. Those issues have been fixed.

For the remote GitHub repository: not yet confirmed. Multiple `git push` attempts and one `git ls-remote origin HEAD` check failed because this machine could not connect to `github.com:443`.

## Release Zip

Release archive generated:

```text
scLifeMamba_release.zip
```

The archive includes README, requirements, environment, source code, scripts, configs, docs, tests, examples, and figures. It excludes raw data, processed data, checkpoints, logs, and outputs.

## Final Status

BLOCKED_WITH_REASON

Reason: local reproducible release is complete and committed, but the commit has not been pushed to the remote GitHub repository due to network connectivity failure.
