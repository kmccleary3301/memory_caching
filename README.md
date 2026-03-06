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
uv run python scripts/reports/release_gate_v1.py --mode repo --out outputs/reports/release_gate_repo_v1.json
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

- CPU-only example:
  - `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
- CUDA example (for CUDA 12.1 builds):
  - `python -m pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Install a `torch` build that matches your local CUDA runtime/driver stack before CUDA workflows.

Install verification:

```bash
mc list-variants
mc smoke-eval --backend linear --device cpu --warmup-steps 1 --batch-size 1 --seq-len 8 --vocab-size 16 --d-model 8 --num-heads 2
```

Debug trace example:

```bash
uv run mc debug-layer --backend linear --aggregation grm --seq-len 8 --d-model 8 --num-heads 2 --out-json outputs/debug/debug_layer.json
```

Onboarding acceptance criteria:

- A new contributor should be able to complete the one-hour onboarding path and open a docs-only PR without changing internal scripts.
- See `docs/CONTRIBUTOR_DRY_RUN.md` for the dry-run record and expected outcomes.

## Key docs

- `docs/reproduction_report.md`
- `docs/CONTRIBUTOR_ONBOARDING.md`
- `docs/CONTRIBUTOR_DRY_RUN.md`
- `docs/CONTRIBUTING.md`
- `docs/ARCHITECTURE.md`
- `docs/ENV_COMPAT_MATRIX.md`
- `docs/CLAIM_TO_EVIDENCE_MATRIX.md`
- `docs/CLAIM_BOUNDARY.md`
- `docs/RELEASE_GATE_CHECKLIST_V1.md`
- `docs/PYPI_RELEASE_RUNBOOK.md`
- `docs/CONSUMER_SUPPORT_MATRIX.md`
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

## Stable package boundary

The stable import surface is intentionally narrow:

- `memory_caching.MCConfig`
- `memory_caching.MemoryCachingLayer`
- `memory_caching.SegmentCache`
- `memory_caching.LinearMemoryBackend`
- `memory_caching.DLABackend`
- `memory_caching.TitansBackend`
- `memory_caching.SWLABackend`

The CLI, smoke helpers, benchmark runners, report generators, and release scripts are repo tooling. They are useful for reproduction work, but they are not the stable public package contract.

Canonical terminology used in this repository:

- `engineering scaffold`: code quality, reproducibility, packaging, and report-generation integrity
- `scientific evidence`: model-backed artifacts with non-smoke targets and truthful manifests
- `paper parity`: faithful reproduction of the paper's reported baselines, metrics, and missing comparison rows

`scientific evidence` is stricter than the engineering scaffold, but it is still not the same as `paper parity`.

For runtime use, prefer the explicit layer methods:

- `layer(x)` for the normal forward path
- `layer.forward_with_cache(x)` when you need cached segment checkpoints
- `layer.inspect(x)` when you need per-token routing/debug rows

## Backend claim boundary

- `linear` is an unnormalized matrix-memory backend used as the wrapper's linear reference path. It should not be read as a full normalized linear-attention baseline.
- `dla`, `titans`, and `swla` are mechanism-oriented reference implementations. They are useful for wrapper-faithfulness work, but they are not yet validated against paper-reported training dynamics or metric parity.
- `titans` and `swla` currently use constant scalar coefficients where the paper presents time-indexed coefficients.
- `soup` is only true state-space mixing for backends that implement state mixing. Non-mixable backends use an explicit output-mixture fallback when that compatibility path is enabled.

## Release surfaces

- Engineering gate:
  - `uv run python scripts/reports/release_gate_v1.py --mode repo --out outputs/reports/release_gate_repo_v1.json`
- Scientific gate:
  - `uv run python scripts/reports/release_gate_v1.py --mode scientific --out outputs/reports/release_gate_scientific_v1.json`

The engineering gate covers repository integrity and public-package mechanics. The scientific gate remains stricter and blocks parity claims unless model-backed evidence and non-smoke targets are present.

What a green scientific gate still does not prove:

- it does not prove full paper parity
- it does not prove missing paper baselines such as `Log-Linear++`
- it does not prove throughput parity or unpublished-author-internal equivalence

## Public API stability

Stable runtime imports:

- `memory_caching.MCConfig`
- `memory_caching.MemoryCachingLayer`
- `memory_caching.SegmentCache`
- `memory_caching.LinearMemoryBackend`
- `memory_caching.DLABackend`
- `memory_caching.TitansBackend`
- `memory_caching.SWLABackend`

Internal or repo-only surfaces:

- `memory_caching.smoke`
- benchmark adapters/runners
- report-generation scripts
- release-gate scripts

## Install from PyPI / wheel / source

From source:

```bash
python -m pip install -e .
```

From source with dev extras:

```bash
python -m pip install -e ".[dev]"
```

From a built wheel:

```bash
python -m pip install dist/*.whl
```

## Minimal examples

- `examples/minimal_layer.py`
- `examples/inspect_layer.py`

Both examples are part of the stable public package surface.

## Remaining paper-parity blocker

Full paper parity is still blocked by missing paper baselines, most notably `configs/train/log_linear_pp.placeholder.yaml`.
