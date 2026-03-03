# Benchmark Protocol

## Supported harnesses

- NIAH (synthetic deterministic)
  - `s_niah_1` passkey retrieval
  - `s_niah_2` number-in-haystack
  - `s_niah_3` uuid-in-haystack
- MQAR (synthetic deterministic)

## Reproducibility knobs

- `seed`
- `samples_per_length` / `samples`
- `context_lengths`
- `num_pairs`
- `num_queries`

## Scoring contract

- Exact-match style scoring for both NIAH and MQAR.
- Metric reported as mean accuracy across generated examples.

## Artifact contract

Each benchmark run writes:

- `metrics.json`
- `manifest.json`

under timestamped directories in `outputs/benchmarks/` (or custom `--out-dir`).

## CLI commands

- `uv run mc bench niah --adapter both --tasks s_niah_1,s_niah_2,s_niah_3 --context-lengths 4096,8192,16384 --samples-per-length 16 --seed 0`
- `uv run mc bench mqar --adapter both --samples 64 --num-pairs 16 --num-queries 4 --seed 0`
