#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/eval outputs/checks outputs/reports outputs/reports/training_telemetry

SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
USE_AMP=${USE_AMP:-1}
USE_COMPILE=${USE_COMPILE:-1}
COMPILE_MODE=${COMPILE_MODE:-max-autotune}
MATMUL_PRECISION=${MATMUL_PRECISION:-high}
DRY_RUN=${DRY_RUN:-0}

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
PILOT_PROFILE=${PILOT_PROFILE:-pilot_full}
MID_PROFILE=${MID_PROFILE:-mid_full}
TARGET_PROFILE=${TARGET_PROFILE:-target_full}
PILOT_MAX_STEPS=${PILOT_MAX_STEPS:-1000}
MID_MAX_STEPS=${MID_MAX_STEPS:-5000}
TARGET_MAX_STEPS=${TARGET_MAX_STEPS:-10000}
PILOT_MAX_SEQ_LEN=${PILOT_MAX_SEQ_LEN:-4096}
MID_MAX_SEQ_LEN=${MID_MAX_SEQ_LEN:-8192}
TARGET_MAX_SEQ_LEN=${TARGET_MAX_SEQ_LEN:-16384}
TRAIN_PARITY_JSON=${TRAIN_PARITY_JSON:-outputs/reports/training_parity_table_full.json}
TRAIN_PARITY_MD=${TRAIN_PARITY_MD:-docs/TRAINING_PARITY_TABLE_FULL.md}
REPORT_SUFFIX=${REPORT_SUFFIX:-}

if [[ "${DRY_RUN}" == "1" ]]; then
  BENCH_ROOT=${BENCH_ROOT:-outputs/benchmarks/full_dataset_dryrun}
  PILOT_PROFILE=${PILOT_PROFILE:-pilot_dryrun}
  MID_PROFILE=${MID_PROFILE:-mid_dryrun}
  TARGET_PROFILE=${TARGET_PROFILE:-target_dryrun}
  PILOT_MAX_STEPS=${PILOT_MAX_STEPS:-2}
  MID_MAX_STEPS=${MID_MAX_STEPS:-2}
  TARGET_MAX_STEPS=${TARGET_MAX_STEPS:-2}
  PILOT_MAX_SEQ_LEN=${PILOT_MAX_SEQ_LEN:-256}
  MID_MAX_SEQ_LEN=${MID_MAX_SEQ_LEN:-256}
  TARGET_MAX_SEQ_LEN=${TARGET_MAX_SEQ_LEN:-256}
  NIAH_CONTEXT_LENGTHS=${NIAH_CONTEXT_LENGTHS:-4096,8192}
  NIAH_SAMPLES_PER_LENGTH=${NIAH_SAMPLES_PER_LENGTH:-4}
  MQAR_SAMPLES=${MQAR_SAMPLES:-16}
  MQAR_PAIR_GRID=${MQAR_PAIR_GRID:-8,16}
  LONG_BENCH_SAMPLES_PER_TASK=${LONG_BENCH_SAMPLES_PER_TASK:-2}
  RETRIEVAL_TRUNCATION_LENGTHS=${RETRIEVAL_TRUNCATION_LENGTHS:-64}
  RETRIEVAL_SAMPLES_PER_DATASET=${RETRIEVAL_SAMPLES_PER_DATASET:-2}
  USE_COMPILE=${USE_COMPILE:-0}
  USE_AMP=${USE_AMP:-0}
  TRAIN_PARITY_JSON=${TRAIN_PARITY_JSON:-outputs/reports/training_parity_table_full_dryrun.json}
  TRAIN_PARITY_MD=${TRAIN_PARITY_MD:-outputs/reports/TRAINING_PARITY_TABLE_FULL_DRYRUN.md}
  REPORT_SUFFIX=${REPORT_SUFFIX:-_dryrun}
fi

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

step_tag() {
  printf "%06d" "$1"
}

run_train_profile "${PILOT_PROFILE}" "configs/train/pilot.yaml" "${PILOT_MAX_STEPS}" "${PILOT_MAX_SEQ_LEN}"
PILOT_STEP_TAG="$(step_tag "${PILOT_MAX_STEPS}")"
uv run python scripts/eval/periodic_eval.py --checkpoint "artifacts/checkpoints/${PILOT_PROFILE}/step_${PILOT_STEP_TAG}.pt" --runner niah --out-json "outputs/eval/${PILOT_PROFILE}_periodic_eval.json"

run_train_profile "${MID_PROFILE}" "configs/train/mid.yaml" "${MID_MAX_STEPS}" "${MID_MAX_SEQ_LEN}"
MID_STEP_TAG="$(step_tag "${MID_MAX_STEPS}")"
uv run python scripts/eval/periodic_eval.py --checkpoint "artifacts/checkpoints/${MID_PROFILE}/step_${MID_STEP_TAG}.pt" --runner niah --out-json "outputs/eval/${MID_PROFILE}_periodic_eval.json"

run_train_profile "${TARGET_PROFILE}" "configs/train/target.yaml" "${TARGET_MAX_STEPS}" "${TARGET_MAX_SEQ_LEN}"
TARGET_STEP_TAG="$(step_tag "${TARGET_MAX_STEPS}")"
uv run python scripts/eval/periodic_eval.py --checkpoint "artifacts/checkpoints/${TARGET_PROFILE}/step_${TARGET_STEP_TAG}.pt" --runner niah --out-json "outputs/eval/${TARGET_PROFILE}_periodic_eval.json"

uv run python scripts/reports/training_parity_table.py \
  --targets-yaml configs/train/paper_targets_full.yaml \
  --checkpoints-root artifacts/checkpoints \
  --eval-root outputs/eval \
  --out-json "${TRAIN_PARITY_JSON}" \
  --out-md "${TRAIN_PARITY_MD}"

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
  --out-json "outputs/reports/full_dataset_benchmark_trend${REPORT_SUFFIX}.json" \
  --out-md "outputs/reports/full_dataset_benchmark_trend${REPORT_SUFFIX}.md"
uv run python scripts/reports/parity_dashboard.py \
  --trend-json "outputs/reports/full_dataset_benchmark_trend${REPORT_SUFFIX}.json" \
  --targets-yaml configs/bench/smoke_targets.yaml \
  --out-json "outputs/reports/full_dataset_parity_dashboard${REPORT_SUFFIX}.json" \
  --out-md "outputs/reports/full_dataset_parity_dashboard${REPORT_SUFFIX}.md"
uv run python scripts/reports/stat_summary.py \
  --root "${BENCH_ROOT}" \
  --out-json "outputs/reports/full_dataset_stat_summary${REPORT_SUFFIX}.json" \
  --out-md "outputs/reports/full_dataset_stat_summary${REPORT_SUFFIX}.md"
uv run python scripts/reports/artifact_checksums.py \
  --path "${BENCH_ROOT}" \
  --path "outputs/reports/full_dataset_benchmark_trend${REPORT_SUFFIX}.json" \
  --path "outputs/reports/full_dataset_parity_dashboard${REPORT_SUFFIX}.json" \
  --path "outputs/reports/full_dataset_stat_summary${REPORT_SUFFIX}.json" \
  --out "outputs/reports/full_dataset_artifact_checksums${REPORT_SUFFIX}.json"

uv run python scripts/reports/write_phase_summary.py \
  --phase "phase3_full_dataset${REPORT_SUFFIX}" \
  --out "outputs/checks/phase3_full_dataset${REPORT_SUFFIX}_summary.json" \
  --input "outputs/reports/full_dataset_benchmark_trend${REPORT_SUFFIX}.json" \
  --input "outputs/reports/full_dataset_parity_dashboard${REPORT_SUFFIX}.json" \
  --input "outputs/reports/full_dataset_stat_summary${REPORT_SUFFIX}.json" \
  --input "outputs/reports/full_dataset_artifact_checksums${REPORT_SUFFIX}.json"
