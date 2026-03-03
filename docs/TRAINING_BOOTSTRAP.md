# Training Bootstrap

1. Train tokenizer scaffold:
- `uv run python scripts/data/train_tokenizer.py --config configs/tokenizer/default.yaml --output artifacts/tokenizer/spm_32000.model`

2. Process data scaffold:
- `uv run python scripts/data/process_data.py --config configs/data/mixture.yaml --tokenizer artifacts/tokenizer/spm_32000.model --output-dir data/processed`

3. Run training scaffold:
- `uv run python scripts/train/train_loop.py --config configs/train/pilot.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/pilot`

4. Trigger periodic eval hook scaffold:
- `uv run python scripts/eval/periodic_eval.py --checkpoint artifacts/checkpoints/pilot/step_000000.pt --runner niah`

5. End-to-end smoke:
- `./scripts/checks/pipeline_smoke.sh`
