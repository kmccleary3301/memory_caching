# Progress Ledger

Date baseline: 2026-03-03

## Weight model

- Phase 0: 10%
- Phase 1: 25%
- Phase 2: 25%
- Phase 3: 20%
- Phase 4: 15%
- Phase 5: 5%

## Computation

Overall progress = sum(phase_weight * phase_completion).

## Update rule

1. Update each phase completion as a percentage.
2. Multiply by phase weight.
3. Sum all weighted contributions.
4. Round to nearest integer for dashboard display.

## Current tracking fields

- `phase_0_completion`
- `phase_1_completion`
- `phase_2_completion`
- `phase_3_completion`
- `phase_4_completion`
- `phase_5_completion`
- `overall_completion`
