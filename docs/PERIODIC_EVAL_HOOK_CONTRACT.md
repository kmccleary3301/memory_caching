# Periodic Eval Hook Contract

The periodic eval hook receives:

- `--checkpoint`: checkpoint artifact path
- `--runner`: benchmark runner id

Expected behavior:

1. resolve checkpoint metadata
2. execute selected benchmark runner
3. emit benchmark artifact bundle
4. register outputs in training/eval logs

Current scaffold entrypoint:

- `scripts/eval/periodic_eval.py`
