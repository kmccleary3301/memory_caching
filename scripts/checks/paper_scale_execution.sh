#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/eval outputs/checks outputs/reports outputs/reports/training_telemetry

SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
USE_AMP=${USE_AMP:-1}
USE_COMPILE=${USE_COMPILE:-1}
COMPILE_MODE=${COMPILE_MODE:-max-autotune}
MATMUL_PRECISION=${MATMUL_PRECISION:-high}

TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-data/processed}
OPTIM_CONFIG=${OPTIM_CONFIG:-configs/optim/schedules.yaml}

BENCH_ROOT=${BENCH_ROOT:-outputs/benchmarks/full_dataset}
NIAH_TASKS=${NIAH_TASKS:-s_niah_1,s_niah_2,s_niah_3}
NIAH_CONTEXT_LENGTHS=${NIAH_CONTEXT_LENGTHS:-4096,8192,16384}
NIAH_SAMPLES_PER_LENGTH=${NIAH_SAMPLES_PER_LENGTH:-16}
MQAR_SAMPLES=${MQAR_SAMPLES:-64}
MQAR_PAIR_GRID=${MQAR_PAIR_GRID:-8,16,32}
MQAR_QUERY_GRID=${MQAR_QUERY_GRID:-2}
LONG_BENCH_TASKS=${LONG_BENCH_TASKS:-single_doc_qa,multi_doc_qa,summarization,few_shot,code}
LONG_BENCH_SAMPLES_PER_TASK=${LONG_BENCH_SAMPLES_PER_TASK:-128}
RETRIEVAL_DATASETS=${RETRIEVAL_DATASETS:-swde,squad,fda}
RETRIEVAL_TRUNCATION_LENGTHS=${RETRIEVAL_TRUNCATION_LENGTHS:-512,1024,2048,16384}
RETRIEVAL_SAMPLES_PER_DATASET=${RETRIEVAL_SAMPLES_PER_DATASET:-128}

LONG_BENCH_DATASET_FILE=${LONG_BENCH_DATASET_FILE:-}
RETRIEVAL_DATASET_FILE=${RETRIEVAL_DATASET_FILE:-}
ALLOW_SUBSET=${ALLOW_SUBSET:-0}

if [[ -z "${LONG_BENCH_DATASET_FILE}" ]]; then
  echo "LONG_BENCH_DATASET_FILE must be set" >&2
  exit 1
fi
if [[ -z "${RETRIEVAL_DATASET_FILE}" ]]; then
  echo "RETRIEVAL_DATASET_FILE must be set" >&2
  exit 1
fi
if [[ ! -f "${LONG_BENCH_DATASET_FILE}" ]]; then
  echo "missing LONG_BENCH_DATASET_FILE: ${LONG_BENCH_DATASET_FILE}" >&2
  exit 1
fi
if [[ ! -f "${RETRIEVAL_DATASET_FILE}" ]]; then
  echo "missing RETRIEVAL_DATASET_FILE: ${RETRIEVAL_DATASET_FILE}" >&2
  exit 1
fi

if [[ "${ALLOW_SUBSET}" != "1" ]]; then
  if [[ "${LONG_BENCH_DATASET_FILE}" == examples/* || "${RETRIEVAL_DATASET_FILE}" == examples/* ]]; then
    echo "subset dataset path detected under examples/. Set ALLOW_SUBSET=1 only for non-paper-scale dry runs." >&2
    exit 1
  fi
fi

compile_args=()
if [[ "${USE_COMPILE}" == "1" ]]; then
  compile_args+=(--compile --compile-mode "${COMPILE_MODE}" --matmul-precision "${MATMUL_PRECISION}")
fi
if [[ "${USE_AMP}" == "1" ]]; then
  compile_args+=(--amp)
fi

run_train_profile() {
  local profile_name="$1"
  local config_path="$2"
  local max_steps="$3"
  local max_seq_len="$4"
  local checkpoint_dir="artifacts/checkpoints/${profile_name}"

  uv run python scripts/train/train_loop.py \
    --config "${config_path}" \
    --optim-config "${OPTIM_CONFIG}" \
    --data-dir "${TRAIN_DATA_DIR}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --max-steps "${max_steps}" \
    --max-seq-len "${max_seq_len}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    "${compile_args[@]}"

  cp artifacts/train_manifest.json "outputs/reports/training_telemetry/${profile_name}_train_manifest.json"
}

run_train_profile "pilot_full" "configs/train/pilot.yaml" "1000" "4096"
uv run python scripts/eval/periodic_eval.py --checkpoint artifacts/checkpoints/pilot_full/step_001000.pt --runner niah --out-json outputs/eval/pilot_full_periodic_eval.json

run_train_profile "mid_full" "configs/train/mid.yaml" "5000" "8192"
uv run python scripts/eval/periodic_eval.py --checkpoint artifacts/checkpoints/mid_full/step_005000.pt --runner niah --out-json outputs/eval/mid_full_periodic_eval.json

run_train_profile "target_full" "configs/train/target.yaml" "10000" "16384"
uv run python scripts/eval/periodic_eval.py --checkpoint artifacts/checkpoints/target_full/step_010000.pt --runner niah --out-json outputs/eval/target_full_periodic_eval.json

uv run python scripts/reports/training_parity_table.py \
  --targets-yaml configs/train/paper_targets_full.yaml \
  --checkpoints-root artifacts/checkpoints \
  --eval-root outputs/eval \
  --out-json outputs/reports/training_parity_table_full.json \
  --out-md docs/TRAINING_PARITY_TABLE_FULL.md

uv run mc bench niah \
  --adapter all \
  --tasks "${NIAH_TASKS}" \
  --context-lengths "${NIAH_CONTEXT_LENGTHS}" \
  --samples-per-length "${NIAH_SAMPLES_PER_LENGTH}" \
  --seed "${SEED}" \
  --out-dir "${BENCH_ROOT}/niah"

uv run mc bench mqar \
  --adapter all \
  --samples "${MQAR_SAMPLES}" \
  --pair-grid "${MQAR_PAIR_GRID}" \
  --query-grid "${MQAR_QUERY_GRID}" \
  --seed "${SEED}" \
  --out-dir "${BENCH_ROOT}/mqar"

uv run mc bench longbench \
  --adapter all \
  --tasks "${LONG_BENCH_TASKS}" \
  --samples-per-task "${LONG_BENCH_SAMPLES_PER_TASK}" \
  --seed "${SEED}" \
  --dataset-file "${LONG_BENCH_DATASET_FILE}" \
  --out-dir "${BENCH_ROOT}/longbench"

uv run mc bench retrieval \
  --adapter all \
  --datasets "${RETRIEVAL_DATASETS}" \
  --truncation-lengths "${RETRIEVAL_TRUNCATION_LENGTHS}" \
  --samples-per-dataset "${RETRIEVAL_SAMPLES_PER_DATASET}" \
  --seed "${SEED}" \
  --dataset-file "${RETRIEVAL_DATASET_FILE}" \
  --out-dir "${BENCH_ROOT}/retrieval"

uv run python scripts/reports/validate_evidence_bundle.py --root "${BENCH_ROOT}"
uv run python scripts/reports/benchmark_trend.py \
  --root "${BENCH_ROOT}" \
  --out-json outputs/reports/full_dataset_benchmark_trend.json \
  --out-md outputs/reports/full_dataset_benchmark_trend.md
uv run python scripts/reports/parity_dashboard.py \
  --trend-json outputs/reports/full_dataset_benchmark_trend.json \
  --targets-yaml configs/bench/paper_targets.yaml \
  --out-json outputs/reports/full_dataset_parity_dashboard.json \
  --out-md outputs/reports/full_dataset_parity_dashboard.md
uv run python scripts/reports/stat_summary.py \
  --root "${BENCH_ROOT}" \
  --out-json outputs/reports/full_dataset_stat_summary.json \
  --out-md outputs/reports/full_dataset_stat_summary.md
uv run python scripts/reports/artifact_checksums.py \
  --path "${BENCH_ROOT}" \
  --path outputs/reports/full_dataset_benchmark_trend.json \
  --path outputs/reports/full_dataset_parity_dashboard.json \
  --path outputs/reports/full_dataset_stat_summary.json \
  --out outputs/reports/full_dataset_artifact_checksums.json

uv run python scripts/reports/write_phase_summary.py \
  --phase phase3_full_dataset \
  --out outputs/checks/phase3_full_dataset_summary.json \
  --input outputs/reports/full_dataset_benchmark_trend.json \
  --input outputs/reports/full_dataset_parity_dashboard.json \
  --input outputs/reports/full_dataset_stat_summary.json \
  --input outputs/reports/full_dataset_artifact_checksums.json
