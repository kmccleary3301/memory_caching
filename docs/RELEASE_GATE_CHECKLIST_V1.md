# Release Gate Checklist v1

## Required checks

- [ ] `uv run python -m pytest -q`
- [ ] `./scripts/checks/phase2.sh`
- [ ] `uv run python scripts/checks/paper_to_code_sync.py`
- [ ] `./scripts/checks/bench_smoke.sh`
- [ ] `./scripts/checks/pipeline_smoke.sh`
- [ ] `./scripts/checks/resume_consistency.sh`
- [ ] `./scripts/checks/independent_repro_pass.sh`
- [ ] `uv run python scripts/reports/validate_evidence_bundle.py --root outputs/benchmarks/phase3_smoke`
- [ ] `uv run python scripts/reports/quarantine_legacy_artifacts.py --root outputs/benchmarks/phase3_smoke --report outputs/reports/quarantine_scan.json`
- [ ] `uv run python scripts/reports/release_gate_v1.py --out outputs/reports/release_gate_v1.json`
- [ ] claim-to-evidence matrix updated
- [ ] reproduction report updated
- [ ] progress ledger updated
- [ ] `docs/paper_to_code_map.yaml` and `docs/PAPER_TO_CODE.md` are synchronized

## Required generated artifacts

- [ ] `outputs/checks/phase2_summary.json`
- [ ] `outputs/checks/phase3_summary.json`
- [ ] `outputs/checks/phase4_summary.json`
- [ ] `outputs/checks/resume_consistency.json`
- [ ] `outputs/reports/phase3_benchmark_trend.json`
- [ ] `outputs/reports/phase3_benchmark_trend.md`
- [ ] `outputs/reports/phase3_parity_dashboard.json`
- [ ] `outputs/reports/phase3_parity_dashboard.md`
- [ ] `outputs/reports/phase3_stat_summary.json`
- [ ] `outputs/reports/phase3_stat_summary.md`
- [ ] `outputs/reports/phase3_artifact_checksums.json`
- [ ] `outputs/reports/training_parity_table.json`
- [ ] `outputs/independent_repro/<stamp>/manifest.json`
- [ ] `outputs/reports/release_gate_v1.json`
- [ ] `outputs/reports/quarantine_scan.json`

## Blocking policy

No external parity claims are allowed unless all checks above are complete.
