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
- Benchmark scoring policy is explicitly task-aligned (`exact_match`, `token_f1`, `rouge_l_f1`) and encoded in row-level outputs.
- LongBench and retrieval runners provide an optional JSONL dataset-file ingestion path.
- Artifact bundles include manifest, metrics, row-level, CSV, and report outputs.
- Release-gate automation emits phase summaries and benchmark trend reports.
- Release-gate checker enforces required evidence files before parity claims.
- Tokenizer/data/training scripts now execute deterministic non-placeholder flows with explicit manifests.
- Training writes real `torch.save` checkpoints and supports resume-path parity checks.
- Benchmark reporting now includes parity dashboard, statistical summary, and checksum archival artifacts.
- Training parity table artifact is generated and gate-enforced.
- Independent clean-environment reproduction pass is implemented, archived, and gate-checked.

## Unsupported claims

- Exact benchmark parity with paper-reported numbers.
- Full-scale distributed training parity.
- Exact unpublished author implementation parity for deep-memory optimizer internals.
- Throughput parity claims against paper systems.

## Blocked claims

- Public full paper-scale metric parity claims remain blocked pending full dataset-scale benchmark execution.

## Progress snapshot

- Phase 0: 100%
- Phase 1: 100%
- Phase 2: 100%
- Phase 3: 100%
- Phase 4: 100%
- Phase 5: 100%
- Overall weighted progress: **100.00%**

## Next milestones

1. Replace benchmark generic adapters with paper-faithful model-backed evaluators.
2. Integrate full LongBench/retrieval datasets and publish parity deltas.
3. Execute scale-oriented training/eval checkpoints and throughput audits on target hardware.
