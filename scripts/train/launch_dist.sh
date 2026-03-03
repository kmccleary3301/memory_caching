#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train/pilot.yaml}
NPROC=${NPROC:-2}

torchrun --nproc_per_node="$NPROC" scripts/train/train_loop.py --config "$CONFIG" --data-dir data/processed --checkpoint-dir artifacts/checkpoints/dist
