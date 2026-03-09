# Reproduction Report

<div align="center">

<a href="../README.md"><img alt="Project" src="https://img.shields.io/badge/project-memory--caching-2088FF"></a>
<a href="./CLAIM_TO_EVIDENCE_MATRIX.md"><img alt="Claim Discipline" src="https://img.shields.io/badge/claims-evidence%20mapped-2ea44f"></a>
<a href="./PAPER_PARITY_BLOCKERS.md"><img alt="Paper Parity" src="https://img.shields.io/badge/paper%20parity-blocked-important"></a>

</div>

Date: 2026-03-06  
Target paper: *Memory Caching: RNNs with Growing Memory* (`arXiv:2602.24281v1`)

This repository currently supports a strong engineering and mechanism-faithfulness
story for the Memory Caching wrapper. It does not yet support a literal claim of
full paper-table parity.

---

## Status Snapshot

| Scope | Status |
|---|---|
| Stable PyPI package surface | `Live` |
| MC wrapper implementation | `Implemented` |
| Engineering gate | `Green` |
| Scientific gate | `Green` |
| Full paper parity | `Blocked` |

---

## What This Repository Supports

### Mechanism-level implementation

- the Memory Caching wrapper is implemented with:
  - segmentation
  - recurrent state update and segment caching
  - RM / GRM / Soup / SSC aggregation paths
- supported backends include:
  - `linear`
  - `dla`
  - `titans`
  - `swla(c=2)`

### Backend and inner-update coverage

- backend-level tests exist for the implemented reference backends
- SWLA(c=2) is covered by paper-equation recurrence/state-mixing tests
- inner-update faithfulness coverage includes:
  - batch/head scaling invariance
  - differentiable-mode temporal gradient-flow coverage
  - explicit Titans update-convention testing

### Reproduction tooling

- benchmark runners exist for:
  - NIAH
  - MQAR
  - LongBench
  - retrieval
- model-backed scientific artifacts, truthful manifests, release gates, and
  artifact reports are in place
- repository-level log-linear work now includes:
  - explicit `LogLinearPP` baseline preset/config family
  - original `LogLinearAttention` reference namespace with dense, recurrent, and correctness-first chunked reference paths

---

## What This Repository Does Not Support

- full paper benchmark parity claims
- throughput parity claims against the paper systems
- exact equivalence with unpublished author internals
- coverage claims for baseline families that are present but not yet fully evaluated for parity

---

## Scientific Evidence Boundaries

| Statement | Interpretation |
|---|---|
| `engineering scaffold` | packaging, tests, artifact plumbing, release mechanics |
| `scientific evidence` | model-backed artifacts with non-smoke targets and truthful manifests |
| `paper parity` | faithful reproduction of the paper's reported baselines, metrics, and missing comparison rows |

Important consequence:

- default rule-based adapters are harness checks, not model-quality evidence
- smoke-target dashboards are repository calibration targets, not paper-reported targets
- a green scientific gate is stricter than the engineering scaffold, but still
  not the same thing as paper parity

---

## Current Evidence Position

| Area | Current state |
|---|---|
| Training scaffold | real checkpoint artifacts written |
| Benchmark path | model-backed path implemented |
| Manifests | truthful train and benchmark manifests |
| Targets | non-smoke scientific targets supported |
| Claim discipline | explicit matrix + boundary docs in place |
| Missing baseline coverage | `LogLinearPP` now present as a baseline preset, but parity evidence remains blocked |

See also:

- [CLAIM_TO_EVIDENCE_MATRIX.md](CLAIM_TO_EVIDENCE_MATRIX.md)
- [CLAIM_BOUNDARY.md](CLAIM_BOUNDARY.md)
- [PAPER_PARITY_BLOCKERS.md](PAPER_PARITY_BLOCKERS.md)

---

## What a Green Scientific Gate Still Does Not Prove

- full paper parity
- `LogLinearPP` is now implemented as a baseline preset, but not yet sufficient for table-level parity claims
- original Guo et al. `LogLinearAttention` remains a separate future implementation target
- throughput parity against the paper's reported systems
- exact equivalence with unpublished author internals

---

## Log-linear Pilot Evidence

The repository now includes a pilot-scale model-backed lane for:

- `LogLinearPP`
- `tiny_loglinear_ref_lm`
- `tiny_loglinear_chunked_lm`

These pilot artifacts are intended to validate integration, truthful manifests,
and claim-safe report generation. They do not establish paper-scale parity for
either the Memory Caching paper or the original LogLinearAttention paper.
