# Engineering Reproduction Scaffold Report

Date: 2026-03-05  
Target paper: *Memory Caching: RNNs with Growing Memory* (arXiv:2602.24281v1)

## Scope statement

This repository is currently an engineering scaffold focused on mechanism-level implementation and validation of the Memory Caching wrapper. It is not a paper-metric parity report.

## What this report supports

- MC wrapper mechanics are implemented and unit-tested:
  - segmentation,
  - recurrent state update and segment caching,
  - RM/GRM/Soup/SSC aggregation paths.
- Linear/DLA/Titans backends are implemented with backend-level tests.
- SWLA(c=2) backend is implemented with backend-level recurrence/state-mixing tests.
- Inner-update faithfulness checks now include:
  - batch/head scaling invariance,
  - differentiable-mode temporal gradient-flow coverage,
  - explicit Titans update-convention testing.
- Benchmark harness/reporting infrastructure is implemented and reproducible as a scaffold.

## What this report does not support

- Full paper benchmark parity claims.
- Throughput parity claims against paper systems.
- Exact equivalence with unpublished author internals.
- Coverage claims for missing paper baselines (Log-Linear++).

## Public claim discipline

- Any benchmark run produced with default rule-based adapters is harness validation, not model-quality evidence.
- Smoke-target dashboards are calibration checks against repository-defined targets, not paper-reported targets.
- Paper-scale claims remain blocked until model-backed full-corpus runs are complete.

## Latest execution evidence (2026-03-05)

- Full `paper_scale_execution.sh` run completed on CUDA through:
  - `pilot_full` (1000 steps),
  - `mid_full` (5000 steps),
  - `target_full` (10000 steps).
- Periodic eval hooks completed for each final checkpoint.
- NIAH, MQAR, LongBench, and retrieval benchmark suites were executed in full-dataset mode.
- Evidence bundles and release gates passed:
  - `validate_evidence_bundle` (full benchmark root),
  - `claim_evidence_lint: PASS`,
  - `release gate v1: PASS`.

## Planner checklist status (2026-03-05)

- In-scope planner checklist items for this execution tranche are complete.
- Remaining items in the broader reproduction roadmap are explicitly outside this closed checklist scope (for example model-backed paper-metric parity work).
