from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "checks" / "claim_evidence_lint.py"


def _run_lint(project_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(project_root)],
        capture_output=True,
        text=True,
    )


def _write_repo_scaffold(root: Path, *, run_generated_evidence_type: str) -> None:
    (root / "configs/bench").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "scripts/checks").mkdir(parents=True, exist_ok=True)
    (root / "scripts/reports").mkdir(parents=True, exist_ok=True)
    (root / "configs/bench/smoke_targets.yaml").write_text("name: smoke\n")
    (root / "src/existing.py").write_text("x = 1\n")
    (root / "scripts/checks/example.sh").write_text("#!/usr/bin/env bash\n")
    (root / "scripts/reports/example.py").write_text("print('ok')\n")
    matrix = f"""# Claim to Evidence Matrix

## Code-backed claims

| claim | evidence_type | location |
|---|---|---|
| Example | unit_test | `src/existing.py` |

## Run-generated claims (CI or local scripts)

| claim | evidence_type | location |
|---|---|---|
| Example run | {run_generated_evidence_type} | `scripts/checks/example.sh`, `scripts/reports/example.py` |

## Blocked claims

| blocked_claim | why_blocked | required_evidence_to_unblock |
|---|---|---|
| x | y | z |
"""
    (root / "docs/CLAIM_TO_EVIDENCE_MATRIX.md").write_text(matrix)


def test_claim_evidence_lint_passes_on_valid_matrix(tmp_path: Path) -> None:
    _write_repo_scaffold(tmp_path, run_generated_evidence_type="generated_evidence")
    proc = _run_lint(tmp_path)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "PASS" in proc.stdout


def test_claim_evidence_lint_fails_when_run_generated_not_tagged(tmp_path: Path) -> None:
    _write_repo_scaffold(tmp_path, run_generated_evidence_type="script_output")
    proc = _run_lint(tmp_path)
    assert proc.returncode != 0
    out = (proc.stderr or "") + (proc.stdout or "")
    assert "run-generated claim row evidence_type must include 'generated'" in out
