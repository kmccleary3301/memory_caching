# Paper-Parity Blockers

Date: 2026-03-06

This document tracks the remaining blockers between the repository's current
state and a literal paper-parity claim for the Memory Caching paper.

## Current status

- Engineering scaffold status: green
- Scientific-gate status: green
- Paper-parity status: blocked

## Active blockers

### 1. Missing paper baselines

- `Log-Linear++` is not implemented in this repository.
- Placeholder tracking file:
  - `configs/train/log_linear_pp.placeholder.yaml`
- Consequence:
  - no full table-level parity claim is currently supportable

### 2. Current scientific runs are tracking runs, not full paper-scale parity runs

- The current scientific artifact set uses truthful manifests, non-smoke
  tracking targets, and model-backed adapters.
- That is enough for scientific-gate integrity.
- It is not enough for a claim that the repository reproduces all paper-reported
  results at full table scale.

### 3. Backend simplifications remain documented but not fully parity-closed

- The linear backend is an unnormalized matrix-memory reference path.
- DLA, Titans, and SWLA still use configured constant coefficients rather than a
  paper-matched learned or time-dependent schedule.
- These are documented implementation choices, but they remain paper-parity
  limitations until explicitly closed.

## Exit criteria for removing this blocker

- Implement `Log-Linear++` as a first-class baseline.
- Add training/eval configs for the missing paper baseline set.
- Re-run full model-backed paper-scale experiments against the relevant tasks.
- Update parity tables and claim-boundary docs to reflect the new evidence.
- Re-review the repository for any remaining backend-level deviations that still
  block literal paper-parity claims.
