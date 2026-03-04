# Backend API Contract

This document defines the required backend interface consumed by `MemoryCachingLayer`.

## Required methods

1. `init_state(batch_size, num_heads, head_dim, device, dtype) -> Any`
2. `update(state, k_t, v_t) -> Any`
3. `apply(state, q_t) -> Tensor`
4. Optional for Soup state-mixing path: `mix_states(states, weights) -> Any`

## Tensor shape contract

- `k_t`, `v_t`, `q_t`: `[B, H, Dh]`
- `apply(...)` output: `[B, H, Dh]`
- `weights` in `mix_states`: `[B, H, S]`

Where:
- `B = batch_size`
- `H = num_heads`
- `Dh = d_model / num_heads`
- `S = number of candidate segment states`

## Runtime enforcement

`MemoryCachingLayer` enforces:
- projected tensor shape checks for `q/k/v/u`
- per-token checks for `k_t/v_t/q_t/u_t`
- non-`None` backend state after `init_state` and `update`
- aggregate output shape/device/dtype checks
- segment context shape checks before caching

## Aggregation capability behavior

- `residual`, `grm`, `ssc`: require `MemoryBackend`
- `soup`:
  - if backend satisfies `MixableMemoryBackend`, use state-space mixing via `mix_states`
  - otherwise fallback to response-space mixing (weighted sum of per-state `apply` outputs)

This fallback is intentional and covered by unit tests.
