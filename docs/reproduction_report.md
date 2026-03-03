# Reproduction Report (Checkpoint)

Date: 2026-03-03
Target paper: *Memory Caching: RNNs with Growing Memory* (arXiv:2602.24281v1, 2026-02-27)

## Scope boundary

This repository currently provides mechanism-level implementation evidence for core Memory Caching operations. It does not yet claim paper-level benchmark parity.

## Supported claims

- Core MC segmentation and cache lifecycle are implemented.
- Aggregators (Residual, GRM, Soup, SSC) are implemented.
- Linear backend is implemented and tested.
- DLA backend prototype is implemented with dot/L2 objective options and stopgrad/differentiable update modes.
- Smoke harness supports both linear and DLA backend paths.
- Synthetic NIAH and MQAR benchmark harnesses are implemented with deterministic generation and exact-match scoring.
- Benchmark artifact outputs include schema-versioned manifest files.

## Unsupported claims

- Exact benchmark parity with paper-reported numbers.
- Full-scale distributed training parity.
- Exact unpublished author implementation parity for deep-memory optimizer internals.
- Throughput parity claims against paper systems.

## Current evidence package

- Unit tests covering segmentation, linear backend math, layer causality/SSC behavior, DLA backend semantics, smoke schema persistence, and benchmark determinism.
- CLI harnesses for smoke and synthetic benchmarks.
- Claim-to-evidence matrix and release gate checklist.

## Known gaps to full reproduction

- Full Titans backend implementation is not yet complete.
- LongBench and retrieval benchmark integrations are not yet complete.
- Paper-scale data/training recipe parity remains open.

## Next milestones

1. Complete Titans backend and parity tests.
2. Add LongBench/retrieval runners with artifact outputs.
3. Add scale-oriented training recipe and evidence reports.
