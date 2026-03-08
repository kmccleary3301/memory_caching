# Memory Caching

<div align="center">

<a href="https://pypi.org/project/memory-caching/"><img alt="PyPI" src="https://img.shields.io/pypi/v/memory-caching?logo=pypi&color=F4B400"></a>
<a href="https://pypi.org/project/memory-caching/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/memory-caching?logo=python&color=3776AB"></a>
<a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-2ea44f"></a>
<a href="https://arxiv.org/abs/2602.24281"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv%3A2602.24281-B31B1B?logo=arxiv"></a>
<a href="https://github.com/kmccleary3301/memory_caching/releases/tag/v0.1.0"><img alt="Release" src="https://img.shields.io/badge/release-v0.1.0-2088FF"></a>

<br><br>

Community PyTorch implementation and reproduction scaffold for
<strong>Memory Caching: RNNs with Growing Memory</strong>.

Installable runtime modules, explicit claim boundaries, model-backed scientific
artifacts, and publication-grade reproduction tooling live in one repository,
with the stable package surface kept intentionally narrow.

</div>

---

## Project Overview

```text
memory_caching/
├── src/memory_caching/       Stable package surface published as memory-caching
│   ├── layer.py              Memory Caching wrapper
│   ├── backends/             Linear, DLA, Titans, SWLA(c=2)
│   ├── bench/                Benchmark adapters, runners, manifests
│   ├── models.py             Tiny model-backed scientific path
│   └── scientific_manifest.py Scientific artifact truthfulness checks
├── configs/                  Train + benchmark + baseline-tracking configs
├── docs/                     Reproduction, release, API, and claim-boundary docs
├── examples/                 Stable public examples
├── scripts/                  Train / eval / gate / packaging entrypoints
└── tests/                    Backend, API, benchmark, and release-path coverage
```

---

## At a Glance

| Area | Current State |
|---|---|
| Stable runtime package | `memory-caching==0.1.0` on PyPI |
| Wrapper mechanisms | Residual / GRM / Soup / SSC |
| Segmentation | Constant and logarithmic |
| Backends | `linear`, `dla`, `titans`, `swla(c=2)` |
| Scientific artifact path | Model-backed, truthful manifests, non-smoke targets |
| Public release status | Publishable package surface with explicit release preflight |
| Full paper parity | Still blocked by incomplete baseline evidence and larger parity gaps |

---

## Project Status

| Scope | Status |
|---|---|
| Stable public PyTorch package | `Active` |
| Mechanism-faithful MC wrapper implementation | `Implemented` |
| Engineering scaffold and packaging integrity | `Green` |
| Scientific gate with model-backed evidence | `Green` |
| Full table-level paper parity | `Blocked by missing baselines` |

This is not official author code. See [reproduction_report.md](docs/reproduction_report.md),
[CLAIM_TO_EVIDENCE_MATRIX.md](docs/CLAIM_TO_EVIDENCE_MATRIX.md), and
[PAPER_PARITY_BLOCKERS.md](docs/PAPER_PARITY_BLOCKERS.md) for the exact
claim surface.

---

## Quickstart

### Option A: `uv` from source

```bash
uv sync --extra dev
uv run mc list-variants
uv run mc smoke-eval --backend linear --device cpu --warmup-steps 1 --batch-size 1 --seq-len 8 --vocab-size 16 --d-model 8 --num-heads 2
```

### Option B: `pip` editable install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
mc list-variants
```

### Option C: PyPI

```bash
python -m pip install memory-caching
python -c "from memory_caching import MCConfig, MemoryCachingLayer; print('ok')"
```

### Torch / CUDA note

- CPU-only example:
  - `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
- CUDA 12.1 example:
  - `python -m pip install torch --index-url https://download.pytorch.org/whl/cu121`

Install a `torch` build that matches your local CUDA runtime and driver stack
before CUDA workflows.

---

## Stable Package Boundary

The supported top-level runtime imports are:

- `memory_caching.MCConfig`
- `memory_caching.MemoryCachingLayer`
- `memory_caching.SegmentCache`
- `memory_caching.LinearMemoryBackend`
- `memory_caching.DLABackend`
- `memory_caching.TitansBackend`
- `memory_caching.SWLABackend`

For runtime use, prefer:

- `layer(x)` for the normal forward path
- `layer.forward_with_cache(x)` when cached segment checkpoints are needed
- `layer.inspect(x)` when per-token routing/debug rows are needed

Repo tooling is intentionally broader than the public package API. CLI wiring,
smoke helpers, benchmark runners, release gates, and report-generation scripts
remain repo-level tooling rather than stable semver-tracked runtime surface.

Full API notes:

- [PUBLIC_API.md](docs/PUBLIC_API.md)

Namespaced experimental/reference modules now present in the package:

- `memory_caching.baselines.LogLinearPP`
- `memory_caching.loglinear.LogLinearAttentionReference`
- `memory_caching.loglinear.ChunkedLogLinearAttentionReference`

---

## Documentation Navigator

### Start here

1. [Documentation Home](docs/README.md)
2. [Reproduction Report](docs/reproduction_report.md)
3. [Public API](docs/PUBLIC_API.md)

### Core docs

| Topic | Link | Purpose |
|---|---|---|
| Documentation index | [docs/README.md](docs/README.md) | Fast entrypoint to the full doc set |
| Reproduction status | [reproduction_report.md](docs/reproduction_report.md) | What is implemented, what is blocked |
| Public runtime API | [PUBLIC_API.md](docs/PUBLIC_API.md) | Stable import surface and boundaries |
| Log-linear terminology | [LOG_LINEAR_TERMINOLOGY.md](docs/LOG_LINEAR_TERMINOLOGY.md) | Separates `LogLinearPP` from original `LogLinearAttention` |
| LogLinearPP baseline | [LOG_LINEAR_PP_BASELINE.md](docs/LOG_LINEAR_PP_BASELINE.md) | MC-paper baseline preset semantics |
| LogLinearAttention reference | [LOG_LINEAR_ATTENTION_REFERENCE.md](docs/LOG_LINEAR_ATTENTION_REFERENCE.md) | Original mechanism reference-path status |
| Architecture | [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layer flow, backend roles, artifact pipeline |
| Claim discipline | [CLAIM_TO_EVIDENCE_MATRIX.md](docs/CLAIM_TO_EVIDENCE_MATRIX.md) | Claim-to-evidence mapping |
| Claim boundaries | [CLAIM_BOUNDARY.md](docs/CLAIM_BOUNDARY.md) | What is explicitly out of claim scope |
| Paper mapping | [PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md) | Paper mechanism to implementation map |
| Progress ledger | [PROGRESS_LEDGER.md](docs/PROGRESS_LEDGER.md) | Current weighted plan state |
| Paper parity blockers | [PAPER_PARITY_BLOCKERS.md](docs/PAPER_PARITY_BLOCKERS.md) | What still blocks literal parity claims |
| Release runbook | [PYPI_RELEASE_RUNBOOK.md](docs/PYPI_RELEASE_RUNBOOK.md) | Package publishing path |
| Support matrix | [CONSUMER_SUPPORT_MATRIX.md](docs/CONSUMER_SUPPORT_MATRIX.md) | User-facing environment support |

---

## Common Workflows

```bash
# List implemented backend/aggregation variants
mc list-variants

# Minimal CPU smoke eval
mc smoke-eval --backend linear --device cpu --warmup-steps 1 --batch-size 1 --seq-len 8 --vocab-size 16 --d-model 8 --num-heads 2

# Debug routing and cache behavior
uv run mc debug-layer --backend linear --aggregation grm --seq-len 8 --d-model 8 --num-heads 2 --out-json outputs/debug/debug_layer.json

# Repository engineering gate
uv run python scripts/reports/release_gate_v1.py --mode repo --out outputs/reports/release_gate_repo_v1.json

# Scientific gate
uv run python scripts/reports/release_gate_v1.py --mode scientific --out outputs/reports/release_gate_scientific_v1.json
```

For dense command coverage, use:

- [BENCHMARK_COMMAND_MATRIX.md](docs/BENCHMARK_COMMAND_MATRIX.md)
- [CONTRIBUTOR_ONBOARDING.md](docs/CONTRIBUTOR_ONBOARDING.md)
- [PYPI_RELEASE_RUNBOOK.md](docs/PYPI_RELEASE_RUNBOOK.md)

---

## Scientific Boundaries

Canonical terminology used in this repository:

- `engineering scaffold`: code quality, reproducibility, packaging, and report-generation integrity
- `scientific evidence`: model-backed artifacts with non-smoke targets and truthful manifests
- `paper parity`: faithful reproduction of the paper's reported baselines, metrics, and missing comparison rows

`scientific evidence` is stricter than the engineering scaffold, but it is still
not the same as `paper parity`.

What a green scientific gate still does not prove:

- full paper parity
- full evaluation evidence for `LogLinearPP`
- original `LogLinearAttention` remains a separate future mechanism track
- throughput parity or unpublished internal-author equivalence

Backend-specific limits also remain important:

- `linear` is an unnormalized matrix-memory reference path, not a normalized
  linear-attention parity claim
- `dla`, `titans`, and `swla` are mechanism-oriented reference implementations
- `titans` and `swla` currently use constant scalar coefficients where the paper
  presents time-indexed coefficients
- `soup` is only true state-space mixing when the backend supports state mixing;
  otherwise the repo uses an explicit output-mixture fallback

---

## Install and Release Surfaces

| Surface | Command | Output |
|---|---|---|
| Editable source install | `python -m pip install -e .` | local runtime package |
| Dev install | `python -m pip install -e ".[dev]"` | local dev + tests + packaging tools |
| Built wheel install | `python -m pip install dist/*.whl` | release-like install path |
| Repo engineering gate | `uv run python scripts/reports/release_gate_v1.py --mode repo ...` | package/repo integrity |
| Scientific gate | `uv run python scripts/reports/release_gate_v1.py --mode scientific ...` | scientific artifact integrity |
| PyPI release preflight | `uv run python scripts/checks/pypi_release_preflight.py` | publish-readiness report |

---

## Examples

Stable published examples:

- [examples/minimal_layer.py](examples/minimal_layer.py)
- [examples/inspect_layer.py](examples/inspect_layer.py)
- [examples/loglinear_reference.py](examples/loglinear_reference.py)
- [examples/loglinear_chunked_reference.py](examples/loglinear_chunked_reference.py)

Current namespaced research/reference surfaces:

- `memory_caching.baselines.LogLinearPP`
- `memory_caching.loglinear.LogLinearAttentionReference`
- `memory_caching.loglinear.ChunkedLogLinearAttentionReference`
- tiny-model families:
  - `tiny_loglinear_ref_lm`
  - `tiny_loglinear_chunked_lm`

Sample subset dataset files included for benchmark dry runs:

- `examples/longbench_subset.jsonl`
- `examples/retrieval_subset.jsonl`

---

## Package and Repository Links

| Resource | Location |
|---|---|
| Paper | [arXiv:2602.24281](https://arxiv.org/abs/2602.24281) |
| PyPI | [pypi.org/project/memory-caching](https://pypi.org/project/memory-caching/) |
| GitHub | [github.com/kmccleary3301/memory_caching](https://github.com/kmccleary3301/memory_caching) |
| Release | [v0.1.0](https://github.com/kmccleary3301/memory_caching/releases/tag/v0.1.0) |

---

## Citation

If you use this repository, cite the original paper and this implementation.

### Original Paper

```bibtex
@article{chandra2026memorycaching,
  title={Memory Caching: RNNs with Growing Memory},
  author={Chandra, ...},
  journal={arXiv preprint arXiv:2602.24281},
  year={2026}
}
```

### This Implementation

```bibtex
@software{memory_caching2026,
  title={memory-caching: Community PyTorch Implementation of Memory Caching},
  author={McCleary, Kyle},
  url={https://github.com/kmccleary3301/memory_caching},
  year={2026}
}
```

---

<div align="center">

Licensed under MIT. Public package surface is documented. Paper-parity limits are
documented explicitly.

</div>
