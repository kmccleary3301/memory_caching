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
- Linear (unnormalized matrix-memory), DLA, and Titans reference backends are implemented with backend-level tests.
- SWLA(c=2) backend is implemented with paper-equation recurrence/state-mixing tests and constant scalar coefficients.
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
- Scientific release claims remain blocked until model-backed full-corpus runs are complete.

## Latest execution evidence (2026-03-05)

- A scaffold execution script (`scripts/checks/paper_scale_execution.sh`, legacy name retained) completed engineering validation runs on CUDA through:
  - `pilot_full` (1000 steps),
  - `mid_full` (5000 steps),
  - `target_full` (10000 steps).
- Periodic eval hooks completed for each final checkpoint in the repository training scaffold.
- Benchmark harnesses were executed in synthetic or dataset-file mode for NIAH, MQAR, LongBench, and retrieval.
- Unless an artifact explicitly records `adapter_type="model_backed"` and non-smoke targets, those outputs should be read as infrastructure checks, not model-quality evidence.
- Evidence bundles and release gates passed:
  - `validate_evidence_bundle` (full benchmark root),
  - `claim_evidence_lint: PASS`,
  - `engineering_release_gate_v1: PASS`.
- Scientific release gate remains blocked pending model-backed evidence and non-smoke targets.

## Checklist status (2026-03-05)

- Closed engineering checklist status: complete.
- Scientific reproduction status: incomplete.
- Remaining blocked work includes:
  - model-backed paper-metric parity work,
  - corrected scientific release gating,
  - missing paper baselines such as Log-Linear++.

## What a green scientific gate still does not prove

- It does not prove full paper parity.
- It does not prove missing paper baselines such as `Log-Linear++`.
- It does not prove throughput parity against the paper's reported systems.
- It does not prove exact equivalence with unpublished author internals.
