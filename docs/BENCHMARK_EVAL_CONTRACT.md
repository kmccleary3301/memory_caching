# Benchmark Evaluation Contract

## Determinism

- All generators and runners must be seed-driven.
- Repeating command + seed must produce identical rows.

## Scoring

- NIAH: normalized exact-match accuracy.
- MQAR: micro and macro accuracy fields.
- LongBench scaffold: task-group accuracy.
- Retrieval scaffold: dataset x truncation accuracy.

## Artifact requirements

Each run must produce:

- `metrics.json`
- `rows.jsonl`
- `summary.csv`
- `report.md`
- `manifest.json`

## Manifest v1 required fields

- schema_version
- run_type
- git_commit
- runner_version
- dataset_revision
- config
- file paths to all produced artifacts
