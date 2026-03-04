#!/usr/bin/env bash
set -euo pipefail

ROOT="artifacts/checkpoints/resume_consistency"
FULL_DIR="${ROOT}/full"
SPLIT_DIR="${ROOT}/split"
RESUME_DIR="${ROOT}/resume"

mkdir -p "${FULL_DIR}" "${SPLIT_DIR}" "${RESUME_DIR}" outputs/checks

uv run python scripts/train/train_loop.py \
  --config configs/train/pilot.yaml \
  --data-dir data/processed \
  --checkpoint-dir "${FULL_DIR}" \
  --seed 123 \
  --max-steps 4 \
  --max-seq-len 128

uv run python scripts/train/train_loop.py \
  --config configs/train/pilot.yaml \
  --data-dir data/processed \
  --checkpoint-dir "${SPLIT_DIR}" \
  --seed 123 \
  --max-steps 2 \
  --max-seq-len 128

uv run python scripts/train/train_loop.py \
  --config configs/train/pilot.yaml \
  --data-dir data/processed \
  --checkpoint-dir "${RESUME_DIR}" \
  --resume-from "${SPLIT_DIR}/step_000002.pt" \
  --seed 123 \
  --max-steps 2 \
  --max-seq-len 128

uv run python scripts/reports/checkpoint_parity.py \
  --expected "${FULL_DIR}/step_000004.pt" \
  --actual "${RESUME_DIR}/step_000004.pt" \
  --tol 1e-7 \
  --out outputs/checks/resume_consistency.json
