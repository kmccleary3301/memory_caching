# Repro Audit Log

## 2026-03-03 M70 execution checkpoint

- Added Titans backend scaffold and tests.
- Added benchmark runner registry + longbench/retrieval scaffolds.
- Added pipeline bootstrap configs/scripts for tokenizer/data/train/eval.
- Added release gate v1, claim boundary doc, evidence scripts.

## 2026-03-03 M90 execution checkpoint

- Hardened backend API contract with runtime shape/state assertions in `MemoryCachingLayer`.
- Added backend contract and reproducibility docs.
- Added phase summary artifact writer and wired phase2/phase3/phase4 check scripts.
- Added benchmark trend report generator and wired bench smoke pipeline.
- Added timeout/retry/resume benchmark sweep orchestrator.
- Added release gate checker script and CI workflow wiring for gate enforcement.
- Added legacy artifact quarantine utility.
- Added optional dataset-file ingestion path for LongBench and retrieval runners.
- Updated progress ledger and reproduction report to 82.55% weighted completion snapshot.

## 2026-03-03 >90 push checkpoint

- Executed `phase2`, `bench_smoke`, `pipeline_smoke`, and release gate checks in order; release gate passed.
- Upgraded tokenizer training from placeholder output to deterministic vocabulary builder with fingerprinted manifests.
- Upgraded data processing from placeholder shard to deterministic weighted tokenized shard generation with source distributions.
- Upgraded training loop from metadata-only writeout to actual tiny-LM optimization with `torch.save` checkpoints and resume support.
- Upgraded periodic eval hook to parse checkpoint payloads and emit proxy metrics.
- Added resume consistency tooling (`scripts/checks/resume_consistency.sh`, `scripts/reports/checkpoint_parity.py`).
- Added committed raw placeholder corpora under `data/raw/` to reduce synthetic fallback reliance.
- Updated contracts/docs/progress to reflect a 90.75% weighted completion snapshot.

## 2026-03-04 parity/statistics hardening checkpoint

- Added benchmark paper-target config at `configs/bench/paper_targets.yaml`.
- Replaced parity dashboard with target-vs-actual delta/status reporting.
- Added statistical summary report generation with mean/std/CI95.
- Added artifact checksum archival report generation.
- Wired parity/statistics/checksum outputs into `bench_smoke` and release gate enforcement.
- Upgraded data processing with deterministic split assignment and split-aware shard outputs.
- Rewrote training loop to include optimizer schedule handling, gradient accumulation, telemetry metrics, and scheduler/RNG checkpoint metadata.
- Added training parity table artifact generation and release-gate requirement.
- Added independent clean-environment repro pass script and archived evidence bundle.
- Updated progress snapshot to 96.25%.
- Added phase-2 parity/guard mode tests and validated `phase2.sh` at 69 passing tests.
- Added explicit model-backed adapter interface and revalidated full bench smoke artifacts.
- Updated progress snapshot to 97.70%.

## 2026-03-04 final plan-completion checkpoint

- Wired unified benchmark scoring helpers into NIAH, MQAR, LongBench, and retrieval runners.
- Added explicit task-group metric routing for LongBench and row-level metric labels in benchmark outputs.
- Added scoring-focused bench tests for MQAR extraction behavior, LongBench metric policy, and retrieval F1 fallback behavior.
- Updated benchmark evaluation contract and claim matrix to reflect task-aligned scoring and dataset-backed scaffold parity evidence.
- Finalized progress ledger and reproduction report to 100.00% plan completion while preserving blocked full paper-scale parity boundaries.

## 2026-03-04 paper-scale execution startup checkpoint

- Added `torch.compile` controls to `scripts/train/train_loop.py` (`--compile`, mode/backend/fullgraph/dynamic, matmul precision metadata).
- Executed full training run matrix on GPU (`RTX 6000 Ada`) with compile+AMP:
  - `pilot_full` (`4096`, `1000` steps)
  - `mid_full` (`8192`, `5000` steps)
  - `target_full` (`16384`, `10000` steps)
- Generated periodic eval artifacts:
  - `outputs/eval/pilot_full_periodic_eval.json`
  - `outputs/eval/mid_full_periodic_eval.json`
  - `outputs/eval/target_full_periodic_eval.json`
- Generated run telemetry/parity artifacts:
  - `outputs/reports/training_telemetry/full_run_summary.json`
  - `outputs/reports/training_parity_table_full.json`
  - `docs/TRAINING_PARITY_TABLE_FULL.md`
- Added paper-scale automation script:
  - `scripts/checks/paper_scale_execution.sh`
- Executed dataset-backed benchmark dry run with local subset corpora:
  - `outputs/benchmarks/full_dataset_dry_run/`
  - `outputs/reports/full_dataset_dry_run_*.json`
- Executed aligned dry-run (`full_dataset_dry_run_v2`) with MQAR query-grid aligned to parity target profile:
  - `outputs/benchmarks/full_dataset_dry_run_v2/`
  - `outputs/reports/full_dataset_dry_run_v2_*.json`
- Added CI artifact guardrails:
  - `scripts/checks/no_large_artifacts.sh`
  - workflow enforcement in `.github/workflows/repro_checks.yml`
