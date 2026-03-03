# Reproduction Report (Checkpoint)

Date: 2026-03-03
Target paper: *Memory Caching: RNNs with Growing Memory* (arXiv:2602.24281v1, 2026-02-27)

## Scope boundary

This repository currently provides mechanism-level implementation evidence for core Memory Caching operations. It does not yet claim paper-level benchmark parity.

## Implemented mechanisms

1. Segmentation and memory caching lifecycle
- Constant and logarithmic segmentation utilities
- Segment cache snapshots with context summaries
- Segment start modes: checkpoint carry and restart

2. Retrieval aggregators
- Residual Memory
- GRM (context-aware gating over segment summaries)
- Memory Soup (state-mixing route when backend supports state mixing)
- SSC (top-k segment routing + online memory inclusion)

3. Backend support
- Linear matrix-valued recurrent memory backend with:
  - outer-product write update
  - query-time retrieval
  - weighted state mixing

4. Smoke execution harness
- Tiny synthetic next-token training loop
- Tiny synthetic eval loop
- Cache statistics reporting for quick integration checks

## Current evidence package

1. Unit tests (implemented)
- Segmentation decomposition and validation
- Linear backend update/apply/mix arithmetic
- Causality guardrail: future-token perturbation does not alter past outputs
- Linear-memory equivalence: GRM and Soup consistency under shared projections

2. Smoke commands (implemented)
- `mc smoke-train`
- `mc smoke-eval`

## Known gaps to full reproduction

1. Deep memory modules are not yet implemented
- DLA-style inner objective updates
- Titans-style memory+optimizer-state updates

2. Benchmark protocol parity is not yet implemented
- NIAH prompt generation + official scoring parity
- LongBench/retrieval task harness and truncation contracts
- MQAR protocol wiring

3. Data/training recipe parity is not yet implemented
- Pretraining corpus composition and tokenizer parity
- Full optimizer schedule parity
- Large-scale distributed training recipe

## Next integration milestones

1. Implement deep-memory backend interface extensions and DLA backend.
2. Implement Titans-style backend with explicit inner-state semantics.
3. Add benchmark runners with evidence outputs (JSON + markdown summaries).
