#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/data/train_tokenizer.py --config configs/tokenizer/default.yaml --output artifacts/tokenizer/spm_32000.model
uv run python scripts/data/process_data.py --config configs/data/mixture.yaml --tokenizer artifacts/tokenizer/spm_32000.model --output-dir data/processed
uv run python scripts/train/train_loop.py --config configs/train/pilot.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/pilot
uv run python scripts/eval/periodic_eval.py --checkpoint artifacts/checkpoints/pilot/step_000000.pt --runner niah
