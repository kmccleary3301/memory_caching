from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pypi_release_preflight_passes_with_explicit_license(tmp_path: Path) -> None:
    out_path = tmp_path / "pypi_release_preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/checks/pypi_release_preflight.py",
            "--out",
            str(out_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "pypi_release_preflight: PASS" in result.stdout
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["checks"]["install_smoke_ok"] is True
    assert report["checks"]["install_smoke_twine_check_ok"] is True
    assert report["checks"]["repo_gate_ok"] is True
    assert report["checks"]["project_license_present"] is True
    assert report["checks"]["root_license_files"] == ["LICENSE"]
