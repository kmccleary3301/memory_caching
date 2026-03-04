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
- Benchmark scoring follows explicit task-aligned policies (`exact_match`, `token_f1`, `rouge_l_f1`) with per-row metric labels.
- Optional JSONL dataset-file ingestion path for LongBench/retrieval runners.
- Included sample dataset files: `examples/longbench_subset.jsonl`, `examples/retrieval_subset.jsonl`.
- Deterministic tokenizer/data/train/eval pipeline with real checkpoint artifacts.
- Artifact bundles: metrics + rows + csv + report + manifest.
- Phase3 reports include trend, parity dashboard, statistical summary, and artifact checksums.

## Quickstart

```bash
uv sync --extra dev
./scripts/checks/phase2.sh
./scripts/checks/bench_smoke.sh
./scripts/checks/pipeline_smoke.sh
./scripts/checks/resume_consistency.sh
uv run python scripts/reports/release_gate_v1.py --out outputs/reports/release_gate_v1.json
```

## Key docs

- `docs/reproduction_report.md`
- `docs/CLAIM_TO_EVIDENCE_MATRIX.md`
- `docs/CLAIM_BOUNDARY.md`
- `docs/RELEASE_GATE_CHECKLIST_V1.md`
- `docs/BACKEND_CAPABILITY_MATRIX.md`
- `docs/BACKEND_API_CONTRACT.md`
- `docs/BACKEND_REPRODUCIBILITY.md`
- `docs/BENCHMARK_COMMAND_MATRIX.md`
- `docs/BENCHMARK_SWEEP_RUNBOOK.md`
- `docs/BENCHMARK_EVAL_CONTRACT.md`
- `docs/TRAINING_PARITY_TABLE.md`
- `docs/TRAINING_BOOTSTRAP.md`
- `docs/PROGRESS_LEDGER.md`

## Notes

- `docs_tmp/` and `outputs/` are intentionally gitignored.
