# Contributor Onboarding (One-Day Path)

## Goal

Get a new contributor from clone to meaningful, safe contribution in a single day without over-claiming faithfulness or parity.

## 1) Environment setup

Recommended (`uv`):

```bash
uv sync --extra dev
```

Alternative (`pip` editable install):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 2) Read-first docs (order matters)

1. `docs/CLAIM_BOUNDARY.md`
2. `docs/CLAIM_TO_EVIDENCE_MATRIX.md`
3. `docs/PAPER_TO_CODE.md`
4. `docs/BACKEND_API_CONTRACT.md`
5. `docs/BACKEND_CAPABILITY_MATRIX.md`

## 3) Fast confidence checks

```bash
./scripts/checks/no_large_artifacts.sh
./scripts/checks/phase2.sh
./scripts/checks/bench_smoke.sh
./scripts/checks/pipeline_smoke.sh
./scripts/checks/resume_consistency.sh
uv run python scripts/reports/release_gate_v1.py --mode repo --out outputs/reports/release_gate_repo_v1.json
```

Notes:

- `phase2.sh` runs `pytest -q`.
- `phase2.sh` also enforces paper-to-code mapping sync via `scripts/checks/paper_to_code_sync.py`.
- Benchmark commands warn when adapters are rule-based compatibility adapters.

## 4) Common contribution categories

1. Backend math/faithfulness (`src/memory_caching/backends/*`, `tests/test_*_backend.py`).
2. MC wrapper behavior (`src/memory_caching/layer.py`, `tests/test_layer.py`).
3. Benchmark/report infrastructure (`src/memory_caching/bench/*`, `scripts/reports/*`).
4. Claim/docs discipline (`docs/*`, `scripts/checks/claim_evidence_lint.py`).

## 5) Rules for safe public claims

- Do not present rule-based benchmark adapter results as model-quality evidence.
- Do not claim paper-metric parity unless evidence is model-backed and reproducible from repository scripts.
- Keep `docs/PAPER_TO_CODE.md` synchronized with `docs/paper_to_code_map.yaml`.

## 6) PR checklist

1. New/changed behavior has tests.
2. `./scripts/checks/phase2.sh` passes.
3. If docs changed, `docs/PAPER_TO_CODE.md` sync check passes.
4. Claim changes are reflected in `docs/CLAIM_TO_EVIDENCE_MATRIX.md` and `docs/reproduction_report.md`.
5. No large artifacts/checkpoints are committed.
