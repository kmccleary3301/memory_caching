#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p artifacts/loglinear_pilots outputs/benchmarks/loglinear_pilots

run_family() {
  local family_name="$1"
  local config_path="$2"
  local artifact_dir="artifacts/loglinear_pilots/${family_name}"
  local checkpoint_dir="${artifact_dir}/checkpoints"
  local checkpoint_path=""

  mkdir -p "$artifact_dir"
  mkdir -p "$checkpoint_dir"

  uv run python scripts/train/train_loop.py \
    --config "$config_path" \
    --data-dir data/processed \
    --checkpoint-dir "$checkpoint_dir"

  checkpoint_path="$(find "$checkpoint_dir" -name 'step_*.pt' | sort | tail -n 1)"
  if [[ -z "$checkpoint_path" ]]; then
    echo "no checkpoint written for ${family_name}" >&2
    exit 1
  fi

  uv run python -m memory_caching.cli bench niah \
    --adapter model \
    --model-checkpoint "$checkpoint_path" \
    --model-device cpu \
    --model-max-new-tokens 1 \
    --model-max-input-tokens 64 \
    --model-seed 123 \
    --out-dir "outputs/benchmarks/loglinear_pilots/${family_name}/niah"

  uv run python -m memory_caching.cli bench mqar \
    --adapter model \
    --model-checkpoint "$checkpoint_path" \
    --model-device cpu \
    --model-max-new-tokens 1 \
    --model-max-input-tokens 64 \
    --model-seed 123 \
    --out-dir "outputs/benchmarks/loglinear_pilots/${family_name}/mqar"
}

run_family "loglinear_pp" "configs/train/loglinear_pp_pilot.yaml"
run_family "tiny_loglinear_ref_lm" "configs/train/loglinear_attention_ref_pilot.yaml"
run_family "tiny_loglinear_chunked_lm" "configs/train/loglinear_attention_chunked_pilot.yaml"

uv run python scripts/reports/loglinear_pilot_report.py
