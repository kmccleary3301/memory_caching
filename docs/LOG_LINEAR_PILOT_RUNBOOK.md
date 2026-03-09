# Log-linear Pilot Runbook

This runbook covers the pilot-scale model-backed lane for the log-linear work.

## Purpose

- produce small pilot artifacts for `LogLinearPP`
- produce small pilot artifacts for `tiny_loglinear_ref_lm`
- produce small pilot artifacts for `tiny_loglinear_chunked_lm`

These artifacts are pilot evidence only. They do not establish paper-scale parity.

## Command

```bash
uv run ./scripts/checks/loglinear_pilot.sh
```

## Outputs

- `artifacts/loglinear_pilots/**`
- `outputs/benchmarks/loglinear_pilots/**`
- `outputs/reports/loglinear_pilot_report.json`
- `outputs/reports/loglinear_pilot_report.md`
