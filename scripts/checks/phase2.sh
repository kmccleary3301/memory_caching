#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/smoke outputs/checks

uv run python -m pytest -q
uv run python scripts/checks/paper_to_code_sync.py
uv run mc smoke-eval --backend linear --device cpu --warmup-steps 2 --out-json outputs/smoke/phase2_linear_eval.json
uv run mc smoke-eval --backend dla --device cpu --warmup-steps 2 --d-model 8 --num-heads 2 --vocab-size 16 --seq-len 8 --batch-size 1 --out-json outputs/smoke/phase2_dla_eval.json
uv run mc smoke-eval --backend titans --device cpu --warmup-steps 2 --d-model 8 --num-heads 2 --vocab-size 16 --seq-len 8 --batch-size 1 --out-json outputs/smoke/phase2_titans_eval.json
uv run mc smoke-eval --backend swla --device cpu --warmup-steps 2 --d-model 8 --num-heads 2 --vocab-size 16 --seq-len 8 --batch-size 1 --out-json outputs/smoke/phase2_swla_eval.json
uv run python scripts/reports/write_phase_summary.py \
  --phase phase2 \
  --out outputs/checks/phase2_summary.json \
  --input outputs/smoke/phase2_linear_eval.json \
  --input outputs/smoke/phase2_dla_eval.json \
  --input outputs/smoke/phase2_titans_eval.json \
  --input outputs/smoke/phase2_swla_eval.json
