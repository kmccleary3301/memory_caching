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

Branch naming:

- `feat/<area>-<short-topic>`
- `fix/<area>-<short-topic>`
- `docs/<short-topic>`

PR title policy:

- Use `<type>: <summary>` where `<type>` is one of `feat`, `fix`, `docs`, `dx`, `refactor`, `test`.
- Include subsystem scope in summary where possible.

PR description required section:

- `Claim impact`: `none`, `docs-only`, or `behavioral`.
- If `behavioral`, explicitly list changed claims and evidence updates.

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
uv run python scripts/reports/release_gate_v1.py --mode repo --out outputs/reports/release_gate_repo_v1.json
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

Onboarding acceptance criteria:

- A new contributor can complete the path in 60 minutes or less on a clean checkout.
- The contributor can identify claim-safe and claim-unsafe wording locations before opening a PR.

## First-hour decision path

- Backend math change:
  - Start: `src/memory_caching/backends/*`, `src/memory_caching/config.py`
  - Tests: `tests/test_*_backend.py`, `tests/test_paper_equations.py`
- Wrapper behavior change:
  - Start: `src/memory_caching/layer.py`
  - Tests: `tests/test_layer.py`, `tests/test_paper_equations.py`
- CLI/check/report change:
  - Start: `src/memory_caching/cli.py`, `scripts/checks/*`, `scripts/reports/*`
  - Tests: `tests/test_cli.py`, script-specific tests
- Docs/claim change:
  - Start: `docs/CLAIM_BOUNDARY.md`, `docs/CLAIM_TO_EVIDENCE_MATRIX.md`, `docs/reproduction_report.md`
  - Check: `scripts/checks/claim_evidence_lint.py`

## Claim safety policy

- Any output produced via default rule-based adapters is harness-validation evidence only.
- Paper parity claims require model-backed evidence and reproducible generation paths.
- Code-backed claims must reference tracked files.

Reviewer checklist for claim-surface changes:

- [ ] Claim matrix updated when claim semantics changed.
- [ ] Reproduction report wording updated for claim scope changes.
- [ ] Evidence path is reproducible from repository scripts.
- [ ] Rule-based adapter outputs are not presented as model-quality evidence.

## Common failure modes

| Failure signature | Likely cause | Fix |
|---|---|---|
| CUDA device unavailable | Incompatible torch/CUDA install | Reinstall torch build matching local runtime/driver |
| `LONG_BENCH_DATASET_FILE must be set` | Missing dataset path env var | Export `LONG_BENCH_DATASET_FILE` and `RETRIEVAL_DATASET_FILE` |
| `docs/PAPER_TO_CODE.md is out of sync` | Mapping file changed without regeneration | Run `uv run python scripts/reports/generate_paper_to_code.py` |
| claim lint header/schema error | Claim matrix columns/section names diverged | Update matrix headers to expected schema and rerun lint |
| config name lint failure on `paper_*.yaml` | Reference-only marker missing | Add `reference_only: true` or rename config |

## Paper-to-code map authoring rules

- Each map item must include either:
  - `symbols` (list of `module::symbol` entries), or
  - `symbol_coverage_reason` (string explaining why symbol mapping is not applicable).
- Every `code` path must exist.
- Use `uv run python scripts/checks/paper_to_code_sync.py` before commit.

## PR checklist (copy/paste)

- [ ] Behavior change has tests.
- [ ] Claim/docs updates are included if claim surface changed.
- [ ] `phase2.sh` passes.
- [ ] `paper_to_code_sync.py` passes.
- [ ] `claim_evidence_lint.py` passes.
- [ ] No large artifacts/checkpoints were committed.
