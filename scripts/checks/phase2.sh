#!/usr/bin/env bash
set -euo pipefail

uv run python -m pytest -q
uv run mc smoke-eval --backend linear --device cpu --warmup-steps 2 --out-json outputs/smoke/phase2_linear_eval.json
uv run mc smoke-eval --backend dla --device cpu --warmup-steps 2 --d-model 8 --num-heads 2 --vocab-size 16 --seq-len 8 --batch-size 1 --out-json outputs/smoke/phase2_dla_eval.json
uv run mc smoke-eval --backend titans --device cpu --warmup-steps 2 --d-model 8 --num-heads 2 --vocab-size 16 --seq-len 8 --batch-size 1 --out-json outputs/smoke/phase2_titans_eval.json
