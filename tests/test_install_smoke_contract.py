from __future__ import annotations

import json
from pathlib import Path


def test_install_smoke_json_uses_clean_venv_contract() -> None:
    path = Path("outputs/checks/install_smoke.json")
    if not path.exists():
        return

    payload = json.loads(path.read_text())
    build_artifacts = payload.get("build_artifacts", {})
    assert isinstance(build_artifacts, dict)
    assert "wheel" in build_artifacts
    assert "sdist" in build_artifacts
    assert payload.get("twine_check_ok") is True
    runs = payload.get("runs", [])
    assert {run.get("mode") for run in runs} == {"wheel", "sdist", "dev"}
    assert all(run.get("verification_env") == "clean_venv" for run in runs)
    assert all("--no-deps" not in str(run.get("install_cmd", "")) for run in runs)
    assert all(run.get("py_typed_ok") is True for run in runs)
    assert all(run.get("import_forward_ok") is True for run in runs)
