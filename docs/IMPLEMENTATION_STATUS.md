# Implementation Status

## Implemented (code present)

- MC configuration schema and validation.
- Segmentation modes: constant and logarithmic.
- MC wrapper with aggregations: residual, GRM, soup, SSC.
- `LogLinearPP` baseline preset/module over the MC wrapper.
- Original `LogLinearAttention` reference namespace with Fenwick helpers, dense oracle, and recurrent reference path.
- Backends: linear matrix-memory, DLA, Titans, SWLA(c=2).
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
- SWLA Eq. (28)-style recurrence and state-mixing equivalence checks.
- Inner-update batch/head invariance checks (DLA/Titans).
- Differentiable inner-update temporal gradient-flow checks (DLA/Titans).
- Rule-based adapter warning and metadata emission in benchmark CLI.

## Implemented with explicit caveats

- Benchmark adapters default to rule-based compatibility adapters unless model-backed adapters are wired.
- The linear backend is an unnormalized matrix-memory backend rather than a full normalized kernel linear-attention baseline.
- DLA, Titans, and SWLA currently use constant scalar coefficients/configured update parameters rather than paper-scale learned or time-dependent schedules.
- `soup` on non-mixable backends requires explicit fallback opt-in (`allow_output_mixture_fallback=true`).
- Smoke-target dashboards are harness checks, not paper-metric parity evidence.

## Not implemented (paper coverage gaps)

- Full chunkwise / optimized original `LogLinearAttention` training path.
- Paper-scale model-backed benchmark parity and throughput parity evidence.
