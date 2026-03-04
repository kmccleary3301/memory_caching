#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/eval outputs/checks

uv run python scripts/data/train_tokenizer.py --config configs/tokenizer/default.yaml --output artifacts/tokenizer/spm_32000.model
uv run python scripts/data/process_data.py --config configs/data/mixture.yaml --tokenizer artifacts/tokenizer/spm_32000.model --output-dir data/processed
uv run python scripts/train/train_loop.py --config configs/train/pilot.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/pilot --max-steps 4 --max-seq-len 128 --seed 0
uv run python scripts/eval/periodic_eval.py --checkpoint artifacts/checkpoints/pilot/step_000000.pt --runner niah --out-json outputs/eval/phase4_periodic_eval.json
uv run python scripts/reports/write_phase_summary.py \
  --phase phase4 \
  --out outputs/checks/phase4_summary.json \
  --input artifacts/tokenizer_manifest.json \
  --input artifacts/data_manifest.json \
  --input artifacts/train_manifest.json \
  --input outputs/eval/phase4_periodic_eval.json
