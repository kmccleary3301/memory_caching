# Training Bootstrap

1. Train tokenizer scaffold:
- `uv run python scripts/data/train_tokenizer.py --config configs/tokenizer/default.yaml --output artifacts/tokenizer/spm_32000.model`

2. Process data scaffold:
- `uv run python scripts/data/process_data.py --config configs/data/mixture.yaml --tokenizer artifacts/tokenizer/spm_32000.model --output-dir data/processed`

3. Run training scaffold:
- `uv run python scripts/train/train_loop.py --config configs/train/pilot.yaml --optim-config configs/optim/schedules.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/pilot --max-steps 4 --max-seq-len 128 --grad-accum-steps 1 --clip-grad-norm 1.0 --deterministic --seed 0`

3b. Resume training scaffold from checkpoint:
- `uv run python scripts/train/train_loop.py --config configs/train/pilot.yaml --optim-config configs/optim/schedules.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/pilot_resume --resume-from artifacts/checkpoints/pilot/step_000000.pt --max-steps 2 --max-seq-len 128 --grad-accum-steps 1 --clip-grad-norm 1.0 --deterministic --seed 0`

4. Trigger periodic eval hook scaffold:
- `uv run python scripts/eval/periodic_eval.py --checkpoint artifacts/checkpoints/pilot/step_000000.pt --runner niah --out-json outputs/eval/phase4_periodic_eval.json`

5. End-to-end smoke:
- `./scripts/checks/pipeline_smoke.sh`

6. Resume determinism check:
- `./scripts/checks/resume_consistency.sh`

7. Full-sequence/full-step execution (compile + AMP):
- `uv run python scripts/train/train_loop.py --config configs/train/pilot.yaml --optim-config configs/optim/schedules.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/pilot_full --max-steps 1000 --max-seq-len 4096 --device cuda --seed 0 --compile --compile-mode max-autotune --matmul-precision high --amp`
- `uv run python scripts/train/train_loop.py --config configs/train/mid.yaml --optim-config configs/optim/schedules.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/mid_full --max-steps 5000 --max-seq-len 8192 --device cuda --seed 0 --compile --compile-mode max-autotune --matmul-precision high --amp`
- `uv run python scripts/train/train_loop.py --config configs/train/target.yaml --optim-config configs/optim/schedules.yaml --data-dir data/processed --checkpoint-dir artifacts/checkpoints/target_full --max-steps 10000 --max-seq-len 16384 --device cuda --seed 0 --compile --compile-mode max-autotune --matmul-precision high --amp`

8. One-shot automation for full execution matrix:
- `LONG_BENCH_DATASET_FILE=/abs/path/longbench_full.jsonl RETRIEVAL_DATASET_FILE=/abs/path/retrieval_full.jsonl ./scripts/checks/paper_scale_execution.sh`
