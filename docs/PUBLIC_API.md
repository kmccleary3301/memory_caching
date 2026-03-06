# Public API

Date: 2026-03-06

This document defines the stable public package surface for the published
`memory-caching` package.

## Stable runtime imports

- `memory_caching.MCConfig`
- `memory_caching.MemoryCachingLayer`
- `memory_caching.SegmentCache`
- `memory_caching.LinearMemoryBackend`
- `memory_caching.DLABackend`
- `memory_caching.TitansBackend`
- `memory_caching.SWLABackend`

## Stability promise

- These imports are the intended semver-tracked runtime surface.
- Renames or removals from this list should be treated as breaking changes.
- New additions to this list should be documented here before release.

## Explicitly non-public surfaces

The following remain repository tooling rather than stable package API:

- `memory_caching.smoke`
- CLI wiring and command entrypoints
- benchmark adapters and benchmark runners
- report-generation scripts
- release-gate and packaging helper scripts

## Examples tied to the public API

- `examples/minimal_layer.py`
- `examples/inspect_layer.py`

These examples are the canonical published examples for the package surface.
