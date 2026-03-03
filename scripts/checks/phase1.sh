#!/usr/bin/env bash
set -euo pipefail

uv run python -m pytest -q
uv run mc smoke-train --steps 10 --device cpu --backend linear --out-json outputs/smoke/linear_train.json
uv run mc smoke-eval --warmup-steps 5 --device cpu --backend linear --out-json outputs/smoke/linear_eval.json
uv run mc smoke-eval --warmup-steps 2 --device cpu --backend dla --d-model 8 --num-heads 2 --vocab-size 16 --seq-len 8 --batch-size 1 --out-json outputs/smoke/dla_eval.json
