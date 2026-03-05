# Memory Caching Reproduction (PyTorch)

Community reproduction of **Memory Caching: RNNs with Growing Memory** (arXiv:2602.24281, Feb 27, 2026).

## Status

- This repository is **not** an official release from the paper authors.
- Current work targets mechanism-faithful implementation of the **Memory Caching wrapper** before paper-scale metric parity.
- We do **not** currently claim exact reproduction of published numbers.

## Current scope

- Core MC wrapper with Residual / GRM / Soup / SSC.
- Segmentation modes: constant and logarithmic.
- Backends: linear, DLA, Titans, SWLA(c=2).
- Smoke harness for train/eval across all backends.
- Benchmark harnesses: NIAH, MQAR, LongBench scaffold, retrieval scaffold.
- Benchmark scoring follows explicit task-aligned policies (`exact_match`, `token_f1`, `rouge_l_f1`) with per-row metric labels.
- Optional JSONL dataset-file ingestion path for LongBench/retrieval runners.
- Included sample dataset files: `examples/longbench_subset.jsonl`, `examples/retrieval_subset.jsonl`.
- Deterministic tokenizer/data/train/eval pipeline with real checkpoint artifacts.
- Artifact bundles: metrics + rows + csv + report + manifest.
- Phase3 reports include trend, smoke-target dashboard, statistical summary, and artifact checksums.
- Default benchmark adapters are **rule-based compatibility adapters**; benchmark scores from these adapters are harness checks, not model-quality evidence.
- Deep-memory backends (DLA/Titans) and SWLA(c=2) are reference implementations and are not yet validated against paper-reported training dynamics or metrics.
- Titans convention note: paper-recursion faithfulness claims require `titans_update_convention="paper"`; `gradient_descent` is provided as an explicit alternative convention.

## Quickstart

`uv` flow (recommended):

```bash
uv sync --extra dev
./scripts/checks/no_large_artifacts.sh
./scripts/checks/phase2.sh
./scripts/checks/bench_smoke.sh
./scripts/checks/pipeline_smoke.sh
./scripts/checks/resume_consistency.sh
uv run python scripts/reports/release_gate_v1.py --out outputs/reports/release_gate_v1.json
```

`pip` editable flow:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
mc list-variants
```

When to use:

- Use `uv` for reproducible local development workflows in this repository.
- Use `pip install -e .` / `pip install -e ".[dev]"` when integrating with an existing Python environment.

Torch/CUDA note:

- Install a `torch` build compatible with your local CUDA runtime/driver stack before running CUDA workflows.

Install verification:

```bash
mc list-variants
uv run mc smoke-eval --backend linear --device cpu --warmup-steps 1
```

Debug trace example:

```bash
uv run mc debug-layer --backend linear --aggregation grm --seq-len 8 --d-model 8 --num-heads 2 --out-json outputs/debug/debug_layer.json
```

## Key docs

- `docs/reproduction_report.md`
- `docs/CONTRIBUTOR_ONBOARDING.md`
- `docs/CONTRIBUTING.md`
- `docs/ARCHITECTURE.md`
- `docs/CLAIM_TO_EVIDENCE_MATRIX.md`
- `docs/CLAIM_BOUNDARY.md`
- `docs/RELEASE_GATE_CHECKLIST_V1.md`
- `docs/BACKEND_CAPABILITY_MATRIX.md`
- `docs/PAPER_TO_CODE.md`
- `docs/BACKEND_API_CONTRACT.md`
- `docs/BACKEND_REPRODUCIBILITY.md`
- `docs/BENCHMARK_COMMAND_MATRIX.md`
- `docs/BENCHMARK_SWEEP_RUNBOOK.md`
- `docs/BENCHMARK_EVAL_CONTRACT.md`
- `docs/TRAINING_PARITY_TABLE.md`
- `docs/TRAINING_PARITY_TABLE_FULL.md`
- `docs/TRAINING_BOOTSTRAP.md`
- `docs/PROGRESS_LEDGER.md`

## Notes

- `docs_tmp/` and `outputs/` are intentionally gitignored.
