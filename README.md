# Memory Caching Reproduction (PyTorch)

Community reproduction of **Memory Caching: RNNs with Growing Memory** (arXiv:2602.24281, Feb 27, 2026).

## Status

- This repository is **not** an official release from the paper authors.
- Current work targets mechanism-faithful implementation before paper-scale metric parity.
- We do **not** currently claim exact reproduction of published numbers.

## Current scope

- Core MC wrapper with Residual / GRM / Soup / SSC.
- Segmentation modes: constant and logarithmic.
- Backends: linear, DLA, Titans.
- Smoke harness for train/eval across all backends.
- Benchmark harnesses: NIAH, MQAR, LongBench scaffold, retrieval scaffold.
- Artifact bundles: metrics + rows + csv + report + manifest.

## Quickstart

```bash
uv sync --extra dev
uv run python -m pytest -q
./scripts/checks/phase2.sh
./scripts/checks/bench_smoke.sh
./scripts/checks/pipeline_smoke.sh
```

## Key docs

- `docs/reproduction_report.md`
- `docs/CLAIM_TO_EVIDENCE_MATRIX.md`
- `docs/CLAIM_BOUNDARY.md`
- `docs/RELEASE_GATE_CHECKLIST_V1.md`
- `docs/BACKEND_CAPABILITY_MATRIX.md`
- `docs/BENCHMARK_COMMAND_MATRIX.md`
- `docs/BENCHMARK_EVAL_CONTRACT.md`
- `docs/TRAINING_BOOTSTRAP.md`
- `docs/PROGRESS_LEDGER.md`

## Notes

- `docs_tmp/` and `outputs/` are intentionally gitignored.
