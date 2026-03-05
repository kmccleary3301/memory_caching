# Implementation Status

## Implemented (code present)

- MC configuration schema and validation.
- Segmentation modes: constant and logarithmic.
- MC wrapper with aggregations: residual, GRM, soup, SSC.
- Backends: linear, DLA, Titans, SWLA(c=2).
- State init modes: checkpoint and restart.
- Benchmark harnesses: NIAH, MQAR, LongBench scaffold, retrieval scaffold.
- CLI commands for smoke and benchmark runs.
- Data/train/report/check scripts for scaffold execution.

## Validated (tracked tests)

- Segmentation determinism and length decomposition.
- MC causality and aggregation invariants.
- Paper-equation analytic invariants (RM/GRM/Soup/SSC + segmentation decomposition).
- Backend contract guards and negative-path validation.
- DLA/Titans objective/update behavior and graph-mode checks.
- SWLA recurrence and state-mixing equivalence checks.
- Inner-update batch/head invariance checks (DLA/Titans).
- Differentiable inner-update temporal gradient-flow checks (DLA/Titans).
- Rule-based adapter warning and metadata emission in benchmark CLI.

## Implemented with explicit caveats

- Benchmark adapters default to rule-based compatibility adapters unless model-backed adapters are wired.
- `soup` on non-mixable backends requires explicit fallback opt-in (`allow_output_mixture_fallback=true`).
- Smoke-target dashboards are harness checks, not paper-metric parity evidence.

## Not implemented (paper coverage gaps)

- Log-Linear++ baseline integration.
- Paper-scale model-backed benchmark parity and throughput parity evidence.
