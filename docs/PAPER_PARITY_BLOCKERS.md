# Paper-Parity Blockers

<div align="center">

<a href="../README.md"><img alt="Project" src="https://img.shields.io/badge/project-memory--caching-2088FF"></a>
<a href="./reproduction_report.md"><img alt="Reproduction" src="https://img.shields.io/badge/reproduction-scaffolded-blue"></a>
<a href="./CLAIM_BOUNDARY.md"><img alt="Parity" src="https://img.shields.io/badge/paper%20parity-blocked-important"></a>

</div>

Date: 2026-03-06

This document tracks the remaining blockers between the repository's current
state and a literal paper-parity claim for the Memory Caching paper.

---

## Current Status

| Area | Status |
|---|---|
| Engineering scaffold | `Green` |
| Scientific gate | `Green` |
| Full paper parity | `Blocked` |

---

## Active Blockers

### 1. Missing paper baselines

| Blocker | Current state | Consequence |
|---|---|---|
| `Log-Linear++` | not implemented | no full table-level parity claim is supportable |

Tracking placeholder:

- `configs/train/log_linear_pp.placeholder.yaml`

### 2. Current scientific runs are tracking runs, not full table-scale parity runs

- the current scientific artifact set uses:
  - truthful manifests
  - non-smoke tracking targets
  - model-backed adapters
- that is enough for scientific-gate integrity
- it is not enough for a claim that all paper-reported results have been
  reproduced at full scale

### 3. Backend simplifications remain documented parity limits

- the linear backend is an unnormalized matrix-memory reference path
- DLA, Titans, and SWLA still use configured constant coefficients rather than
  a paper-matched learned or time-dependent schedule
- these are documented implementation choices, but they remain parity limits
  until explicitly closed

---

## Exit Criteria

- implement `Log-Linear++` as a first-class baseline
- add training and evaluation configs for the missing paper baseline set
- re-run full model-backed paper-scale experiments against the relevant tasks
- update parity tables and claim-boundary docs with the new evidence
- re-review remaining backend deviations before making any literal parity claim
