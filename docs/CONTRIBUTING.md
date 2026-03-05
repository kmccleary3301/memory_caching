# Contributing

## Scope and standards

- Keep claims aligned with `docs/CLAIM_BOUNDARY.md`.
- Do not treat rule-based benchmark adapter scores as model-quality evidence.
- Prefer small, reviewable commits with clear acceptance criteria.

## Branch, commit, and PR workflow

1. Create a feature branch from `main`.
2. Implement with tests/docs in the same branch.
3. Run required checks.
4. Open PR with problem statement, change summary, and claim impact.

Commit style:

- Use imperative subject lines.
- Include scope prefixes when helpful (`feat:`, `fix:`, `docs:`, `dx:`).

## Required checks before PR

```bash
./scripts/checks/no_large_artifacts.sh
./scripts/checks/phase2.sh
./scripts/checks/bench_smoke.sh
./scripts/checks/pipeline_smoke.sh
./scripts/checks/resume_consistency.sh
uv run python scripts/checks/paper_to_code_sync.py
uv run python scripts/checks/claim_evidence_lint.py
uv run python scripts/reports/release_gate_v1.py --out outputs/reports/release_gate_v1.json
```

## 60-minute new-contributor path

1. Install environment with `uv sync --extra dev` or `pip install -e ".[dev]"`.
2. Run `mc list-variants`.
3. Run `./scripts/checks/phase2.sh`.
4. Make a docs-only change in `docs/`.
5. Run `uv run python scripts/checks/claim_evidence_lint.py`.
6. Open PR.

Expected confidence signal:

- A contributor can execute the above without modifying internal scripts and without ambiguity about claim boundaries.

## Claim safety policy

- Any output produced via default rule-based adapters is harness-validation evidence only.
- Paper parity claims require model-backed evidence and reproducible generation paths.
- Code-backed claims must reference tracked files.

## Common failure modes

- CUDA mismatch: verify `torch` build and runtime CUDA compatibility.
- Dataset path mismatch: ensure `LONG_BENCH_DATASET_FILE` and `RETRIEVAL_DATASET_FILE` are set correctly.
- Mapping drift: run `uv run python scripts/reports/generate_paper_to_code.py`.
- Claim matrix lint failure: ensure section headers/columns match expected schema.

## PR checklist (copy/paste)

- [ ] Behavior change has tests.
- [ ] Claim/docs updates are included if claim surface changed.
- [ ] `phase2.sh` passes.
- [ ] `paper_to_code_sync.py` passes.
- [ ] `claim_evidence_lint.py` passes.
- [ ] No large artifacts/checkpoints were committed.
