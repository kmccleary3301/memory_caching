# Titans Backend Notes

## Scope

This implementation is a mechanism-focused Titans-style deep memory backend for Memory Caching experiments.

## Implemented semantics

- `TitansState` with memory parameters and optimizer-like state (`S`).
- Inner objectives:
  - `l2`
  - `dot`
- Update modes:
  - `stopgrad`
  - `differentiable`
- Explicit update conventions:
  - `paper`: `S_t = beta * S_{t-1} - eta * grad_t`, `M_t = alpha * M_{t-1} - S_t`
  - `gradient_descent`: `S_t = beta * S_{t-1} + eta * grad_t`, `M_t = alpha * M_{t-1} - S_t`
- Retention coefficient (`alpha`) and momentum coefficient (`beta`).
- State mixing support for Soup aggregation.

## Sign-convention rationale

- `paper` mode exists to mirror the written recurrence form as directly as possible.
- `gradient_descent` mode preserves the practical descent-style behavior many users expect from optimizer-state updates.
- Both are first-class, test-covered conventions; faithfulness claims to the written paper recursion require selecting `update_convention="paper"`.

## Differentiable-mode semantics

- `inner_update_mode="differentiable"` preserves graph connectivity through inner updates so later-token losses can backpropagate through earlier inner steps.
- `inner_update_mode="stopgrad"` explicitly detaches the inner update state each step.
- This setting changes gradient-flow semantics, not just speed/memory.

## Caveats

- This is not claimed to be bit-level identical to unpublished author code.
- Throughput/perf tuning is intentionally deferred.
- Paper-level parity claims remain blocked until benchmark and scale evidence is complete.
