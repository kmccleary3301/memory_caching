# Implementation Status

## Completed

- MC configuration schema and validation.
- Segment cache and backend protocol contracts.
- Segmentation utilities (constant/logarithmic + explicit lengths validation).
- Linear backend (update/apply/mix).
- DLA backend prototype:
  - dot/L2 inner objectives
  - stopgrad/differentiable update modes
  - state mixing support for Soup
- MC layer aggregators:
  - Residual
  - GRM
  - Soup
  - SSC
- State init modes:
  - checkpoint
  - restart
- CLI commands:
  - status
  - list-variants
  - segment
  - smoke-train
  - smoke-eval
  - bench niah
  - bench mqar
- Synthetic benchmark harnesses:
  - deterministic NIAH generators and scoring
  - deterministic MQAR generators and scoring
  - artifact bundle writer (`metrics.json` + `manifest.json`)
- Expanded test suite for layer, DLA, smoke, and benchmark determinism.

## Remaining

- Titans backend implementation.
- LongBench/retrieval benchmark integration.
- Paper-scale data/training recipe parity.
- Throughput and large-scale parity evidence.
