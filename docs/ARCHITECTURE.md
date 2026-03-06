# Architecture

## Documentation index

- Claim boundaries:
  - `docs/CLAIM_BOUNDARY.md`
  - `docs/CLAIM_TO_EVIDENCE_MATRIX.md`
- Reproduction/report status:
  - `docs/reproduction_report.md`
  - `docs/PROGRESS_LEDGER.md`
- Gates/checklists:
  - `docs/RELEASE_GATE_CHECKLIST_V1.md`
  - `scripts/reports/release_gate_v1.py`
  - `scripts/checks/claim_evidence_lint.py`
  - `scripts/checks/paper_to_code_sync.py`

## Core flow

1. Input tensor `x` is projected to `q/k/v/u`.
2. Sequence is segmented by constant or logarithmic policy.
3. Backend state updates recurrently per token.
4. Segment-end state/context snapshots are cached.
5. Aggregation combines cached + online responses using RM/GRM/Soup/SSC.
6. Output heads are merged and projected with `o_proj`.

Primary implementation: `src/memory_caching/layer.py`.

## Backend contract

Required methods:

1. `init_state(...)`
2. `update(state, k_t, v_t)`
3. `apply(state, q_t)`

Optional:

1. `mix_states(states, weights)` for Soup state mixing.

Contract details: `docs/BACKEND_API_CONTRACT.md`.

## Backends

- `linear`: matrix memory update/apply.
- `dla`: deep memory with inner objective update.
- `titans`: deep memory with explicit update conventions.
- `swla`: second-order recurrence backend (`c=2` path).

Capability matrix: `docs/BACKEND_CAPABILITY_MATRIX.md`.

## Aggregations

- `residual` (RM): additive cached+online responses.
- `grm`: context-weighted response mixture.
- `soup`: state-space mixing when backend supports `mix_states`.
- `ssc`: top-k selective cached routing + online path.

## Bench/report pipeline map

1. Benchmark runners: `src/memory_caching/bench/*`.
2. Artifact bundling: `src/memory_caching/bench/artifacts.py`.
3. Report generation: `scripts/reports/*`.
4. Gate/lints:
   - `scripts/checks/config_name_lint.py`
   - `scripts/checks/paper_to_code_sync.py`
   - `scripts/checks/claim_evidence_lint.py`
   - `scripts/reports/release_gate_v1.py`

## Model-faithfulness boundaries

- Wrapper and backend mechanics are test-backed implementation claims.
- Rule-based adapter benchmarks are harness checks, not model-quality validation.
- Paper-metric parity remains blocked unless model-backed evidence is present.

## Public package surface vs repo tooling

The stable package surface is intentionally smaller than the repo surface.

Stable import surface:

- `memory_caching.MCConfig`
- `memory_caching.MemoryCachingLayer`
- `memory_caching.SegmentCache`
- `memory_caching.LinearMemoryBackend`
- `memory_caching.DLABackend`
- `memory_caching.TitansBackend`
- `memory_caching.SWLABackend`

Repo-only or research tooling surface:

- `memory_caching.smoke`
- CLI commands under `mc`
- benchmark adapters and runners
- report generation and release-gate scripts

`MemoryCachingLayer.forward()` exposes only the runtime tensor path. Cache retrieval and debug inspection use explicit side-channel methods: `forward_with_cache()` and `inspect()`.

## Backend claim boundaries

- `linear` is the wrapper's unnormalized matrix-memory reference backend, not a full normalized linear-attention baseline.
- `dla`, `titans`, and `swla` are implemented as mechanism-oriented reference backends with test-backed wrapper integration.
- `titans` and `swla` currently use constant scalar coefficients where the paper presents time-indexed coefficients.
- `swla` is implemented as the current `c=2` recurrence path with previous-outer-product carry, and should be described as such.
- Backends remain lightweight protocol objects rather than `nn.Module` subclasses for now. That is a deliberate package-boundary choice, not an assertion that the API is final.
