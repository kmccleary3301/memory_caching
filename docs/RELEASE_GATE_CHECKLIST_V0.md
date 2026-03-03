# Release Gate Checklist v0

## Required

- [ ] `uv run python -m pytest -q` passes
- [ ] `uv run mc smoke-train ...` writes metrics JSON
- [ ] `uv run mc smoke-eval ...` writes metrics JSON
- [ ] `uv run mc bench niah ...` writes metrics + manifest
- [ ] `uv run mc bench mqar ...` writes metrics + manifest
- [ ] claim-to-evidence matrix is updated
- [ ] reproduction report supported/unsupported claims updated
- [ ] progress ledger refreshed with current percentages

## Blocking policy

No public parity or benchmark claims should be made if any required item is unchecked.
