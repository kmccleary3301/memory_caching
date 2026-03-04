# Benchmark Sweep Runbook

## Goal

Run all benchmark families with retry/timeout/resume support and collect sweep metadata.

## Command

```bash
uv run python scripts/bench/run_benchmark_sweep.py \
  --root outputs/benchmarks/sweeps/default \
  --retries 1 \
  --timeout-sec 1800 \
  --report outputs/reports/benchmark_sweep_report.json
```

## Resume behavior

- Each benchmark family writes a marker under:
  - `outputs/benchmarks/sweeps/default/_markers/*.done.json`
- Re-running the same command skips completed steps.
- Use `--force` to rerun completed steps.

## Post-run checks

1. `uv run python scripts/reports/validate_evidence_bundle.py --root outputs/benchmarks/sweeps/default`
2. `uv run python scripts/reports/benchmark_trend.py --root outputs/benchmarks/sweeps/default --out-json outputs/reports/sweep_trend.json --out-md outputs/reports/sweep_trend.md`
