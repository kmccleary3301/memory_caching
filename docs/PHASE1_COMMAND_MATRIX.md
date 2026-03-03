# Phase 1 Command Matrix

## Test command

- Command: `uv run python -m pytest -q`
- Expected: all tests pass, no import errors, no collection failures.

## Smoke commands

- Train (linear):
  - Command: `uv run mc smoke-train --steps 20 --device cpu --backend linear --out-json outputs/smoke/linear_train.json`
- Eval (linear):
  - Command: `uv run mc smoke-eval --warmup-steps 10 --device cpu --backend linear --out-json outputs/smoke/linear_eval.json`
- Eval (dla):
  - Command: `uv run mc smoke-eval --warmup-steps 2 --device cpu --backend dla --d-model 8 --num-heads 2 --vocab-size 16 --seq-len 8 --batch-size 1 --out-json outputs/smoke/dla_eval.json`

## Expected smoke JSON keys

- `mode`
- `device`
- `backend`
- `steps`
- `batch_size`
- `seq_len`
- `vocab_size`
- `initial_loss`
- `final_loss`
- `eval_loss`
- `eval_accuracy`
- `cache_segments`
- `mean_segment_len`
- `trainable_params`
