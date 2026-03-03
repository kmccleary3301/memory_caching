# DLA Backend Notes

## Scope

This DLA backend is a mechanism-focused implementation for the memory-caching wrapper.

## Implemented semantics

- Memory state as a small per-head MLP parameter tensor set.
- Inner update objective choices:
  - `dot`
  - `l2`
- Inner update modes:
  - `stopgrad` (default practical mode)
  - `differentiable` (debug/ablation mode)
- Optional momentum in inner updates.
- State mixing support for Soup path.

## Non-goals in current implementation

- Exact paper-kernel parity with unpublished author code.
- Throughput-optimized inner-loop kernels.
- Full-scale distributed memory-state optimization stack.

## Known divergence risks

- Inner-loop optimizer details may differ from paper internals.
- Training-scale stability characteristics are not yet tuned.
- Benchmark parity claims are out of scope until benchmark harness and scale runs complete.
