# Progress Ledger

Date: 2026-03-05

## Weight model

- Phase 0: 10%
- Phase 1: 25%
- Phase 2: 25%
- Phase 3: 20%
- Phase 4: 15%
- Phase 5: 5%

## Current phase completions

- phase_0_completion: 100%
- phase_1_completion: 100%
- phase_2_completion: 100%
- phase_3_completion: 100%
- phase_4_completion: 100%
- phase_5_completion: 100%

## Weighted computation

Overall =

- 0.10 * 1.00
- 0.25 * 1.00
- 0.25 * 1.00
- 0.20 * 1.00
- 0.15 * 1.00
- 0.05 * 1.00

Total: **100.00%**

## Latest milestone update (2026-03-05)

- Full paper-scale execution script completed on CUDA for `pilot_full`, `mid_full`, and `target_full`.
- Full benchmark/report chain completed (NIAH, MQAR, LongBench, retrieval).
- Final integrity and release checks passed:
  - `claim_evidence_lint: PASS`
  - `release gate v1: PASS`

## Update rule

1. Update each phase completion.
2. Use `scripts/reports/update_progress.py`.
3. Publish updated bars in this file and `docs/reproduction_report.md`.
