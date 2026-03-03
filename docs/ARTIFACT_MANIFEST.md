# Artifact Manifest Format (v1)

## File

`manifest.json`

## Required fields

- `schema_version` (string)
- `run_type` (string)
- `utc_timestamp` (ISO-8601 string)
- `git_commit` (string)
- `config` (object)
- `metrics_file` (string path)

## Example

```json
{
  "schema_version": "v1",
  "run_type": "niah",
  "utc_timestamp": "2026-03-03T22:05:00Z",
  "git_commit": "<sha>",
  "config": {
    "adapter": "both",
    "tasks": ["s_niah_1", "s_niah_2", "s_niah_3"],
    "context_lengths": [4096, 8192, 16384],
    "samples_per_length": 16,
    "seed": 0
  },
  "metrics_file": "outputs/benchmarks/<stamp>/metrics.json"
}
```
