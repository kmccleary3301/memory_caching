# Release Gate Checklist v1

## Required checks

- [ ] `uv run python -m pytest -q`
- [ ] `./scripts/checks/phase2.sh`
- [ ] `./scripts/checks/bench_smoke.sh`
- [ ] `./scripts/checks/pipeline_smoke.sh`
- [ ] `uv run python scripts/reports/validate_evidence_bundle.py --root outputs/benchmarks`
- [ ] claim-to-evidence matrix updated
- [ ] reproduction report updated
- [ ] progress ledger updated

## Blocking policy

No external parity claims are allowed unless all checks above are complete.
