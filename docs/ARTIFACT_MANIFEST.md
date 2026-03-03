# Artifact Manifest Format (v1)

## File

`manifest.json`

## Required fields

- `schema_version` (string)
- `run_type` (string)
- `utc_timestamp` (ISO-8601 string)
- `git_commit` (string)
- `runner_version` (string)
- `dataset_revision` (string)
- `config` (object)
- `metrics_file` (string path)
- `rows_file` (string path)
- `summary_csv_file` (string path)
- `report_file` (string path)

## Example

```json
{
  "schema_version": "v1",
  "run_type": "niah",
  "utc_timestamp": "2026-03-03T22:05:00Z",
  "git_commit": "<sha>",
  "runner_version": "v0.2",
  "dataset_revision": "synthetic-v2",
  "config": {
    "adapter": "all",
    "tasks": ["s_niah_1", "s_niah_2", "s_niah_3"],
    "context_lengths": [4096, 8192, 16384],
    "samples_per_length": 16,
    "seed": 0
  },
  "metrics_file": "outputs/benchmarks/<stamp>/metrics.json",
  "rows_file": "outputs/benchmarks/<stamp>/rows.jsonl",
  "summary_csv_file": "outputs/benchmarks/<stamp>/summary.csv",
  "report_file": "outputs/benchmarks/<stamp>/report.md"
}
```
