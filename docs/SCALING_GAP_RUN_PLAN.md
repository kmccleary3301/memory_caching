# Scaling Gap Run Plan

This document enumerates the remaining runs required to close paper-scale parity gaps.

## Current bounded scope

- Mechanism-level MC behavior: implemented and release-gated.
- Benchmark harness + evidence contract: implemented and release-gated.
- Deterministic tokenizer/data/train/eval scaffolds: implemented with resume parity.

## Open scaling gaps

1. Full-sequence training length parity
- Current smoke execution uses `--max-seq-len 128`.
- Required: full profile lengths (`4096`, `8192`, `16384`) with sustained runs.

2. Full-step training parity
- Current archive runs use `--max-steps 4` for pilot/mid/target sanity passes.
- Required: execute profile target steps:
  - pilot: `1000`
  - mid: `5000`
  - target: `10000`

3. Throughput parity
- Current telemetry records local smoke rates only.
- Required: run on target hardware and publish:
  - sustained `tokens/s`
  - step-time distribution
  - memory utilization curves
  - effective cost/throughput comparison

4. Dataset-backed benchmark parity at paper scale
- Current longbench/retrieval dataset-file mode is operational on subsets.
- Required: full benchmark corpus ingestion with official split fidelity and scoring parity report.

## Required execution matrix

1. `pilot_full`
- config: `configs/train/pilot.yaml`
- constraints: full seq len, full steps
- required outputs:
  - `artifacts/checkpoints/pilot_full/`
  - `outputs/eval/pilot_full_periodic_eval.json`
  - throughput telemetry extract

2. `mid_full`
- config: `configs/train/mid.yaml`
- constraints: full seq len, full steps
- required outputs:
  - `artifacts/checkpoints/mid_full/`
  - `outputs/eval/mid_full_periodic_eval.json`
  - throughput telemetry extract

3. `target_full`
- config: `configs/train/target.yaml`
- constraints: full seq len, full steps
- required outputs:
  - `artifacts/checkpoints/target_full/`
  - `outputs/eval/target_full_periodic_eval.json`
  - throughput telemetry extract

4. `bench_full_dataset`
- benchmark root: `outputs/benchmarks/full_dataset/`
- required outputs:
  - trend/parity/stat/checksum report set
  - claim matrix update with exact unblocked claims

## Automation entrypoint

- Full execution script:
  - `scripts/checks/paper_scale_execution.sh`
- Required dataset environment:
  - `LONG_BENCH_DATASET_FILE=/absolute/path/to/longbench_full.jsonl`
  - `RETRIEVAL_DATASET_FILE=/absolute/path/to/retrieval_full.jsonl`
- Safety:
  - script fails if dataset paths point into `examples/` unless `ALLOW_SUBSET=1`.

## Current execution status (2026-03-04)

- Training matrix completed on GPU with compile+AMP:
  - `pilot_full` -> `artifacts/checkpoints/pilot_full/step_001000.pt`
  - `mid_full` -> `artifacts/checkpoints/mid_full/step_005000.pt`
  - `target_full` -> `artifacts/checkpoints/target_full/step_010000.pt`
- Training telemetry and parity artifacts generated:
  - `outputs/reports/training_telemetry/full_run_summary.json`
  - `outputs/reports/training_parity_table_full.json`
  - `docs/TRAINING_PARITY_TABLE_FULL.md`
- Dataset-backed benchmark dry-run generated from local subset corpora:
  - `outputs/benchmarks/full_dataset_dry_run/`
  - `outputs/reports/full_dataset_dry_run_benchmark_trend.json`
  - `outputs/reports/full_dataset_dry_run_parity_dashboard.json`
  - `outputs/reports/full_dataset_dry_run_stat_summary.json`
  - `outputs/reports/full_dataset_dry_run_artifact_checksums.json`
- Remaining blocker for full paper-scale benchmark parity:
  - full LongBench + retrieval corpus files are not present in repository-local data paths.

## Exit criteria for 100%

- All four execution matrix runs complete and archived.
- Release gate passes on full-dataset/full-scale evidence.
- Claim boundary and reproduction report updated to reflect final parity status.
