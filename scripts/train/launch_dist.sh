#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train/pilot.yaml}
NPROC=${NPROC:-2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-artifacts/checkpoints/dist}
SEED=${SEED:-0}
MAX_STEPS=${MAX_STEPS:-4}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}
OPTIM_CONFIG=${OPTIM_CONFIG:-configs/optim/schedules.yaml}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR
export MASTER_PORT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED="$SEED"
export TORCH_NCCL_BLOCKING_WAIT=1

torchrun \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/train/train_loop.py \
  --config "$CONFIG" \
  --optim-config "$OPTIM_CONFIG" \
  --data-dir data/processed \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --seed "$SEED" \
  --max-steps "$MAX_STEPS" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --deterministic
