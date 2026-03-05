## Summary

- What changed:
- Why:

## Claim impact

- [ ] `none`
- [ ] `docs-only`
- [ ] `behavioral` (list changed claims below)

Changed claims/evidence:

-

## Required checks

- [ ] `./scripts/checks/no_large_artifacts.sh`
- [ ] `./scripts/checks/phase2.sh`
- [ ] `uv run python scripts/checks/paper_to_code_sync.py`
- [ ] `uv run python scripts/checks/claim_evidence_lint.py`
- [ ] `uv run python scripts/reports/release_gate_v1.py --out outputs/reports/release_gate_v1.json`

## Claim-boundary acknowledgment

- [ ] I did not present rule-based benchmark adapter output as model-quality evidence.
- [ ] I updated claim/docs files when behavior or claim surface changed.
