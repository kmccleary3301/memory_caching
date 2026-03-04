# Reproduction Report (Checkpoint)

Date: 2026-03-03
Target paper: *Memory Caching: RNNs with Growing Memory* (arXiv:2602.24281v1, 2026-02-27)

## Scope boundary

This repository provides mechanism-level implementation evidence for core Memory Caching operations and scaffold benchmark infrastructure. It does not yet claim paper-level metric parity.

## Supported claims

- Core MC segmentation and cache lifecycle are implemented.
- Aggregators (Residual, GRM, Soup, SSC) are implemented.
- Backends implemented: linear, DLA, Titans.
- Smoke harness supports linear, DLA, and Titans paths with schema-stable metrics.
- NIAH and MQAR deterministic harnesses are implemented.
- LongBench and retrieval scaffold runners are implemented with artifact output contracts.
- LongBench and retrieval runners provide an optional JSONL dataset-file ingestion path.
- Artifact bundles include manifest, metrics, row-level, CSV, and report outputs.
- Release-gate automation emits phase summaries and benchmark trend reports.
- Release-gate checker enforces required evidence files before parity claims.
- Tokenizer/data/training scripts now execute deterministic non-placeholder flows with explicit manifests.
- Training writes real `torch.save` checkpoints and supports resume-path parity checks.

## Unsupported claims

- Exact benchmark parity with paper-reported numbers.
- Full-scale distributed training parity.
- Exact unpublished author implementation parity for deep-memory optimizer internals.
- Throughput parity claims against paper systems.

## Blocked claims

- Public full-reproduction claims are blocked until release gate v1 is fully green with dataset-backed benchmark evidence.

## Progress snapshot

- Phase 0: 100%
- Phase 1: 100%
- Phase 2: 94%
- Phase 3: 87%
- Phase 4: 75%
- Phase 5: 72%
- Overall weighted progress: **90.75%**

## Next milestones

1. Replace benchmark generic adapters with paper-faithful model-backed evaluators.
2. Integrate full LongBench/retrieval datasets and publish parity deltas.
3. Execute scale-oriented training/eval checkpoints and throughput audits on target hardware.
