# SWLA (c=2) Implementation Plan

## Status snapshot

- Backend recurrence/update/apply/mix implementation: completed.
- Config/smoke/CLI wiring: completed.
- SWLA backend tests and layer-level GRM-vs-Soup invariant tests: completed in code.
- Claim promotion to fully unblocked paper-parity claim: pending (requires full model-backed paper-scale evidence).

## Goal

Add a paper-aligned SWLA backend so Memory Caching coverage includes the proof-of-concept architecture family referenced in the paper.

## Scope for P1

- Implement SWLA(c=2) backend with explicit recurrence and retrieval path.
- Add unit tests for:
  - recurrence correctness against explicit reference computation,
  - GRM vs Soup equivalence for linear-memory assumptions,
  - batch/head invariance.
- Add config + smoke wiring without changing default backend selection.
- Add explicit claim boundary updates once tests pass.

## Proposed phases

1. Backend completion
- Promote scaffold in `src/memory_caching/backends/swla.py` to full backend with validated coefficients/schedules.
- Decide coefficient policy:
  - constants only (initial),
  - optional token-dependent coefficients (follow-up).

2. Integration
- Extend backend enum/config parsing to include `swla`.
- Wire into smoke model backend factory and CLI backend options.

3. Test and compliance
- Add `tests/test_swla_backend.py`.
- Extend `tests/test_layer.py` with SWLA-specific invariants.
- Update `docs/PAPER_TO_CODE.md` mapping with SWLA rows.

4. Claim updates
- Move SWLA claim from blocked to code-backed in claim matrix only after tests and scripts pass.

## Non-goals for this phase

- Full Log-Linear++ implementation.
- Paper-scale performance parity claims.
