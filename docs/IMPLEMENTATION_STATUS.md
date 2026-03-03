# Implementation Status

## Completed

- MC configuration schema and validation.
- Segment cache and backend protocol contracts.
- Segmentation utilities (constant/logarithmic + explicit lengths validation).
- Backends:
  - linear
  - dla
  - titans
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
  - bench list/niah/mqar/longbench/retrieval
- Synthetic benchmark harnesses:
  - deterministic NIAH generators + scoring
  - deterministic MQAR generators + micro/macro scoring
  - LongBench scaffold runner
  - retrieval scaffold runner
  - artifact bundle writer (`metrics.json`, `rows.jsonl`, `summary.csv`, `report.md`, `manifest.json`)
- Bootstrap pipeline scaffolds:
  - tokenizer config + script
  - data mixture config + processing script
  - train profile configs + train loop scaffold
  - periodic eval hook scaffold
- Evidence and governance scaffolds:
  - release gate v1
  - claim boundary
  - claim-to-evidence matrix with blocked claims
  - progress updater and evidence validators

## Remaining

- Dataset-backed LongBench/retrieval implementations for parity claims.
- Full paper-scale training and throughput parity evidence.
