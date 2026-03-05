# Environment Compatibility Matrix

## Supported baseline

- Python: `>=3.10,<3.13`
- Package manager: `uv` (recommended) or `pip`
- OS: Linux/macOS (primary workflows)

## Torch runtime guidance

| Use case | Torch wheel channel example | Notes |
|---|---|---|
| CPU-only | `https://download.pytorch.org/whl/cpu` | Good for lint/tests/smoke checks without GPU |
| CUDA 12.1 | `https://download.pytorch.org/whl/cu121` | Requires compatible NVIDIA driver/runtime |

## Compatibility policy

- Match torch wheel CUDA variant to your local runtime/driver.
- If CUDA is unavailable or mismatched, use CPU mode for checks and smoke workflows.
- Do not claim cross-platform bitwise determinism across AMP/compile/device variants.

## Quick verification

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
mc smoke-eval --backend linear --device cpu --warmup-steps 1 --batch-size 1 --seq-len 8 --vocab-size 16 --d-model 8 --num-heads 2
```
