# Release Gate Checklist v1

This repository now uses two gate surfaces:

- `engineering_release_gate_v1`: repository integrity, packaging, installability, docs/evidence consistency.
- `scientific_release_gate_v1`: everything in the engineering gate, plus eligibility for public scientific/parity claims.

## Engineering gate

Command:

- [ ] `uv run python scripts/reports/release_gate_v1.py --mode repo --out outputs/reports/release_gate_repo_v1.json`

Required checks:

- [ ] `uv run python -m pytest -q`
- [ ] `./scripts/checks/phase2.sh`
- [ ] `uv run python scripts/checks/config_name_lint.py`
- [ ] `uv run python scripts/checks/paper_to_code_sync.py`
- [ ] `./scripts/checks/install_smoke.sh`
- [ ] `./scripts/checks/bench_smoke.sh`
- [ ] `./scripts/checks/pipeline_smoke.sh`
- [ ] `./scripts/checks/resume_consistency.sh`
- [ ] `./scripts/checks/independent_repro_pass.sh`
- [ ] `uv run python scripts/reports/validate_evidence_bundle.py --root outputs/benchmarks/phase3_smoke`
- [ ] `uv run python scripts/reports/quarantine_legacy_artifacts.py --root outputs/benchmarks/phase3_smoke --report outputs/reports/quarantine_scan.json`
- [ ] claim-to-evidence matrix updated
- [ ] reproduction report updated
- [ ] progress ledger updated
- [ ] `docs/paper_to_code_map.yaml` and `docs/PAPER_TO_CODE.md` are synchronized

Required generated artifacts:

- [ ] `dist/*.whl`
- [ ] `dist/*.tar.gz`
- [ ] `outputs/checks/phase2_summary.json`
- [ ] `outputs/checks/phase3_summary.json`
- [ ] `outputs/checks/phase4_summary.json`
- [ ] `outputs/checks/resume_consistency.json`
- [ ] `outputs/checks/install_smoke.json`
- [ ] `outputs/checks/install_smoke_wheel_eval.json`
- [ ] `outputs/checks/install_smoke_sdist_eval.json`
- [ ] `outputs/checks/install_smoke_dev_eval.json`
- [ ] `outputs/reports/phase3_benchmark_trend.json`
- [ ] `outputs/reports/phase3_benchmark_trend.md`
- [ ] `outputs/reports/phase3_parity_dashboard.json`
- [ ] `outputs/reports/phase3_parity_dashboard.md`
- [ ] `outputs/reports/phase3_stat_summary.json`
- [ ] `outputs/reports/phase3_stat_summary.md`
- [ ] `outputs/reports/phase3_artifact_checksums.json`
- [ ] `outputs/reports/training_parity_table.json`
- [ ] `outputs/independent_repro/<stamp>/manifest.json`
- [ ] `outputs/reports/release_gate_repo_v1.json`
- [ ] `outputs/reports/quarantine_scan.json`

Install smoke artifact schema:

- `generated_at_utc`: UTC timestamp string.
- `ok`: boolean.
- `build_artifacts`: object with `wheel` and `sdist` relative paths.
- `runs`: list with three entries (`wheel`, `sdist`, `dev`) and fields:
  - `mode`
  - `artifact_kind`
  - `install_cmd`
  - `eval_artifact`

## Scientific gate

Command:

- [ ] `uv run python scripts/reports/release_gate_v1.py --mode scientific --out outputs/reports/release_gate_scientific_v1.json`

Additional scientific requirements beyond the engineering gate:

- [ ] `artifacts/train_manifest.json` exists and records a non-scaffold model family
- [ ] `artifacts/train_manifest.json` records `uses_memory_caching=true`
- [ ] `outputs/reports/full_dataset_parity_dashboard.json` does not use `smoke_targets.yaml`
- [ ] benchmark manifests and metrics record `adapter_type="model_backed"`
- [ ] no blocked scientific claim remains in claim-boundary documentation

Required generated artifact:

- [ ] `outputs/reports/release_gate_scientific_v1.json`

## Blocking policy

- No external engineering release statement should cite the scientific gate unless `outputs/reports/release_gate_scientific_v1.json` is green.
- No external scientific/parity claim is allowed unless the scientific gate is green.
