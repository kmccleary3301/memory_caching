# Memory Caching Reproduction (PyTorch)

Community reproduction of **Memory Caching: RNNs with Growing Memory** (arXiv:2602.24281, Feb 27, 2026).

## Status

- This repository is **not** an official release from the paper authors.
- Current phase targets **mechanism-faithful implementation** before paper-scale metric parity.
- We do **not** currently claim exact reproduction of published numbers.

## Scope (initial)

- Implement a reusable Memory Caching (MC) wrapper with:
  - Residual Memory
  - Gated Residual Memory (GRM)
  - Memory Soup
  - Sparse Selective Caching (SSC)
- Support segmentation modes:
  - Constant segment length
  - Logarithmic segmentation
- Integrate with recurrent memory backends in phases:
  - Phase 1: linear memory baseline
  - Phase 2: deep-memory backends (DLA, Titans-style updates)

## Repository layout

- `docs/PHASE_1_PLAN.md`: concrete implementation plan and acceptance checks
- `docs/PAPER_ARTIFACT_PIN.md`: local artifact and paper pinning notes
- `src/memory_caching/`: implementation package

## Quickstart

```bash
uv sync --extra dev
uv run python -m memory_caching.cli status
```

## Notes

- `docs_tmp/` is intentionally gitignored and excluded from version control.
