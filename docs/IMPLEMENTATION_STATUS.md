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
- Benchmark execution hardening:
  - benchmark sweep script with timeout/retry/resume markers
  - benchmark trend report generator (`json` + `md`)
  - stricter evidence manifest validator
  - optional JSONL dataset-file ingestion for longbench/retrieval + sample files
- Bootstrap pipeline scaffolds:
  - tokenizer config + deterministic vocabulary trainer (JSON model + vocab file + corpus fingerprint)
  - data mixture config + deterministic weighted sampling + tokenized shard writer
  - train profile configs + actual tiny-LM train loop (`torch.save` checkpoints + resume support)
  - periodic eval hook reads checkpoint loss tails and emits proxy score
  - resume consistency parity checker (`resume` vs `full` checkpoint tensor diff)
- Phase summary artifacts:
  - phase2 summary writer
  - phase3 summary writer
  - phase4 summary writer
- Evidence and governance scaffolds:
  - release gate v1
  - release gate checker script
  - claim boundary
  - claim-to-evidence matrix with blocked claims
  - progress updater and evidence validators
  - legacy artifact quarantine utility
  - backend API contract and reproducibility notes

## Remaining

- Production dataset ingestion/execution for LongBench/retrieval parity claims at paper scale (dataset-file mode exists, paper dataset integration/runbooks pending).
- Full paper-scale training throughput/efficiency parity evidence on target hardware.
