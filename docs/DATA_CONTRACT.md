# Data Contract

## Source provenance requirements

- Each source must declare:
  - name
  - uri/path
  - version or revision marker
  - weight in mixture
  - split policy ratios (`train/val/test`)

## Processing manifest requirements

Data processing must emit manifest entries for:

- input config path
- tokenizer artifact path
- output shard directory
- generation timestamp
- shard index file path
- source distribution counts
- total/average token statistics
- dataset fingerprint hash

## Local override policy

- Local overrides are allowed for development.
- Overrides must be explicitly declared in config and artifacts.

## Record schema (`data/processed/shard_*.jsonl`)

Each record contains:

- `record_id` (int)
- `source` (string)
- `split` (`train|val|test`)
- `text` (string)
- `token_ids` (list[int])
- `token_count` (int)

Split output files:

- training shards: `shard_*.jsonl`
- validation shards: `val_shard_*.jsonl`
- test shards: `test_shard_*.jsonl`
