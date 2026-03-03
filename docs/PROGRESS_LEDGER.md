# Progress Ledger

Date: 2026-03-03

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
- phase_2_completion: 85%
- phase_3_completion: 70%
- phase_4_completion: 10%
- phase_5_completion: 35%

## Weighted computation

Overall =

- 0.10 * 1.00
- 0.25 * 1.00
- 0.25 * 0.85
- 0.20 * 0.70
- 0.15 * 0.10
- 0.05 * 0.35

Total: **73.50%**

## Update rule

1. Update each phase completion.
2. Use `scripts/reports/update_progress.py`.
3. Publish updated bars in this file and `docs/reproduction_report.md`.
