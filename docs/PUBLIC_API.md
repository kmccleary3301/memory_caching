# Public API

<div align="center">

<a href="../README.md"><img alt="Package" src="https://img.shields.io/badge/package-memory--caching-2088FF"></a>
<a href="https://pypi.org/project/memory-caching/"><img alt="PyPI" src="https://img.shields.io/pypi/v/memory-caching?logo=pypi&color=F4B400"></a>
<a href="./PYPI_RELEASE_RUNBOOK.md"><img alt="Release" src="https://img.shields.io/badge/release-preflight%20clean-2ea44f"></a>

</div>

Date: 2026-03-06

This document defines the stable runtime surface for the published
`memory-caching` package.

---

## Stable Runtime Imports

| Import | Role |
|---|---|
| `memory_caching.MCConfig` | wrapper configuration |
| `memory_caching.MemoryCachingLayer` | primary runtime module |
| `memory_caching.SegmentCache` | cached segment record |
| `memory_caching.LinearMemoryBackend` | linear reference backend |
| `memory_caching.DLABackend` | DLA reference backend |
| `memory_caching.TitansBackend` | Titans reference backend |
| `memory_caching.SWLABackend` | SWLA(c=2) reference backend |

---

## Stability Promise

- imports listed above are the intended semver-tracked runtime surface
- renames or removals from this list should be treated as breaking changes
- additions to this list should be documented here before release

---

## Preferred Runtime Entry Points

| Pattern | Use case |
|---|---|
| `layer(x)` | normal forward path |
| `layer.forward_with_cache(x)` | return cached segment checkpoints |
| `layer.inspect(x)` | return per-token routing/debug rows |

---

## Explicitly Non-Public Surfaces

These remain repository tooling rather than stable package API:

- `memory_caching.smoke`
- CLI wiring and command entrypoints
- benchmark adapters and benchmark runners
- report-generation scripts
- release-gate and packaging helper scripts

---

## Namespaced Reference Surfaces

The package currently includes namespaced reference/research modules that are
not yet part of the stable top-level API:

- `memory_caching.baselines.LogLinearPP`
- `memory_caching.loglinear.LogLinearAttentionReference`
- `memory_caching.loglinear.ChunkedLogLinearAttentionReference`

These modules are present for research and parity work, but their current
stability promise is weaker than the top-level imports listed above.

---

## Canonical Published Examples

- `examples/minimal_layer.py`
- `examples/inspect_layer.py`

These examples are the published reference examples for the public package
surface.
