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
- Momentum-style state update and retention coefficient (`alpha`).
- State mixing support for Soup aggregation.

## Caveats

- This is not claimed to be bit-level identical to unpublished author code.
- Throughput/perf tuning is intentionally deferred.
- Paper-level parity claims remain blocked until benchmark and scale evidence is complete.
