# Architecture

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
