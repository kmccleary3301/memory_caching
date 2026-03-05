# Backend Reproducibility Notes

## Scope

This note describes what is currently reproducible for backend behavior and what is still blocked.

## Deterministic components

- Fixed-seed smoke harnesses for `linear`, `dla`, `titans`, and `swla`.
- Deterministic synthetic benchmark generators (NIAH, MQAR).
- Deterministic artifact schemas with machine-readable manifests.
- Runtime backend contract checks in `MemoryCachingLayer`.

## Evidence hooks

- Phase 2 summary artifact: `outputs/checks/phase2_summary.json`
- Smoke metrics:
  - `outputs/smoke/phase2_linear_eval.json`
  - `outputs/smoke/phase2_dla_eval.json`
  - `outputs/smoke/phase2_titans_eval.json`
  - `outputs/smoke/phase2_swla_eval.json`
- Unit tests:
  - `tests/test_linear_backend.py`
  - `tests/test_dla_backend.py`
  - `tests/test_titans_backend.py`
  - `tests/test_swla_backend.py`
  - `tests/test_layer.py`

## Current limitations

- LongBench/retrieval parity remains scaffold-level until dataset-backed evaluators are wired.
- Throughput parity and hardware-normalized claims are blocked pending dedicated perf harnesses.
- Paper-scale training parity is not yet claimed.

## Claim policy

Use backend-level claims only for mechanism correctness and contract stability, not for full paper parity.
