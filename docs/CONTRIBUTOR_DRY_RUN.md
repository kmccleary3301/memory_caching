# Contributor Dry Run Record

Date: 2026-03-05

## Objective

Validate that a new contributor can complete the one-hour onboarding path and open a low-risk PR without ambiguity on claim boundaries.

## Dry-run path executed

1. Environment setup:
   - `uv sync --extra dev`
2. Basic sanity:
   - `mc list-variants`
3. Mechanism checks:
   - `./scripts/checks/phase2.sh`
4. Docs-only change simulation:
   - edit a docs file and run lint checks
5. Claim discipline checks:
   - `uv run python scripts/checks/claim_evidence_lint.py`
6. PR checklist review:
   - `docs/CONTRIBUTING.md` checklist completed

## Expected outcomes

- Contributor can identify where to make:
  - backend changes,
  - wrapper changes,
  - CLI/check/report changes,
  - claim/docs changes.
- Contributor can run required checks without modifying internal scripts.
- Contributor understands that rule-based benchmark outputs are harness evidence, not model-quality evidence.

## Acceptance outcome

- Onboarding path is considered complete when the contributor can submit a docs-only PR with all required checks passing and no claim-boundary violations.
