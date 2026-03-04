# Training Bootstrap

1. Train tokenizer scaffold:
- `uv run python scripts/data/train_tokenizer.py --config configs/tokenizer/default.yaml --output artifacts/tokenizer/spm_32000.model`

2. Process data scaffold:
- `uv run python scripts/data/process_data.py --config configs/data/mixture.yaml --tokenizer artifacts/tokenizer/spm_32000.model --output-dir data/processed`

3. Run training scaffold:
- `uv run python scripts/train/train_loop.py --config configs/train/pilot.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/pilot --max-steps 4 --max-seq-len 128 --seed 0`

3b. Resume training scaffold from checkpoint:
- `uv run python scripts/train/train_loop.py --config configs/train/pilot.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/pilot_resume --resume-from artifacts/checkpoints/pilot/step_000000.pt --max-steps 2 --max-seq-len 128 --seed 0`

4. Trigger periodic eval hook scaffold:
- `uv run python scripts/eval/periodic_eval.py --checkpoint artifacts/checkpoints/pilot/step_000000.pt --runner niah --out-json outputs/eval/phase4_periodic_eval.json`

5. End-to-end smoke:
- `./scripts/checks/pipeline_smoke.sh`

6. Resume determinism check:
- `./scripts/checks/resume_consistency.sh`
