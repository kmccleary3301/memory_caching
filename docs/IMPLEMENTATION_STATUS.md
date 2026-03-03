# Implementation Status

## Phase 1

### Completed

- MC configuration schema and validation
- Segment cache and backend protocol contracts
- Segmentation utilities:
  - constant-size segmentation
  - logarithmic segmentation
  - explicit segment-length validation
- Linear memory backend:
  - matrix-valued update and retrieval
  - state mixing operation for Soup path
- MC layer with aggregators:
  - Residual
  - GRM
  - Memory Soup
  - SSC
- State init modes:
  - checkpoint carry across segments
  - restart per segment
- CLI commands:
  - status
  - variant listing
  - segment inspection
  - smoke-train
  - smoke-eval
- Focused test suite:
  - segmentation determinism
  - linear backend arithmetic
  - causality/no-future-leak check
  - linear backend GRM-vs-Soup equivalence

### Remaining in Phase 1

- Extend smoke harness into a paper-oriented benchmark runner contract
- Add reproducibility artifacts for benchmark prompt templates and scoring

## Out of scope (current phase)

- Paper-scale training and exact metric parity
- Throughput-optimized kernels
- Full deep-memory backend parity (DLA/Titans)
