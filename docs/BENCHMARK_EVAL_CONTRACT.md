# Benchmark Evaluation Contract

## Determinism

- All generators and runners must be seed-driven.
- Repeating command + seed must produce identical rows.

## Scoring

- NIAH: normalized exact-match accuracy.
- MQAR: micro and macro accuracy fields.
- LongBench scaffold: task-group accuracy.
- Retrieval scaffold: dataset x truncation accuracy.

## Dataset-backed mode (LongBench/Retrieval)

- LongBench accepts `--dataset-file` (JSONL with `task_group`, `prompt`, `answer` or `answers`).
- Retrieval accepts `--dataset-file` (JSONL with `dataset`, `document|context`, `question`, `answer|answers`).
- If `--dataset-file` is provided, rows are sampled deterministically by seed.
- If `--dataset-file` is omitted, synthetic scaffold prompts are used.

## Artifact requirements

Each run must produce:

- `metrics.json`
- `rows.jsonl`
- `summary.csv`
- `report.md`
- `manifest.json`

Additional reporting artifacts required for phase gate:

- `phase3_benchmark_trend.json`
- `phase3_parity_dashboard.json`
- `phase3_stat_summary.json`
- `phase3_artifact_checksums.json`

## Manifest v1 required fields

- schema_version
- run_type
- git_commit
- runner_version
- dataset_revision
- config
- file paths to all produced artifacts
