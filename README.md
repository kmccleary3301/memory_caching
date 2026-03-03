# Memory Caching Reproduction (PyTorch)

Community reproduction of **Memory Caching: RNNs with Growing Memory** (arXiv:2602.24281, Feb 27, 2026).

## Status

- This repository is **not** an official release from the paper authors.
- Current work targets mechanism-faithful implementation before paper-scale metric parity.
- We do **not** currently claim exact reproduction of published numbers.

## Current scope

- Core MC wrapper with Residual / GRM / Soup / SSC.
- Segmentation modes: constant and logarithmic.
- Backends: linear matrix memory + DLA prototype backend.
- Smoke harness for train/eval (linear and DLA paths).
- Synthetic benchmark harness: NIAH + MQAR with artifact manifests.

## Quickstart

```bash
uv sync --extra dev
uv run python -m pytest -q
uv run mc smoke-train --steps 20 --device cpu --backend linear
uv run mc smoke-eval --warmup-steps 2 --device cpu --backend dla --d-model 8 --num-heads 2 --vocab-size 16 --seq-len 8 --batch-size 1
uv run mc bench niah --adapter both --tasks s_niah_1,s_niah_2,s_niah_3 --context-lengths 4096,8192 --samples-per-length 8
uv run mc bench mqar --adapter both --samples 32 --num-pairs 16 --num-queries 4
```

## Project docs

- `docs/PHASE_1_PLAN.md`
- `docs/IMPLEMENTATION_STATUS.md`
- `docs/reproduction_report.md`
- `docs/CLAIM_TO_EVIDENCE_MATRIX.md`
- `docs/BENCHMARK_PROTOCOL.md`
- `docs/RELEASE_GATE_CHECKLIST_V0.md`
- `docs/PROGRESS_LEDGER.md`
- `docs/ARTIFACT_MANIFEST.md`
- `docs/PAPER_ARTIFACT_PIN.md`

## Notes

- `docs_tmp/` is intentionally gitignored and excluded from version control.
