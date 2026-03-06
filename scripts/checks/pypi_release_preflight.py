from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
INSTALL_SMOKE = ROOT / "outputs" / "checks" / "install_smoke.json"
REPO_GATE = ROOT / "outputs" / "reports" / "release_gate_repo_v1.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _root_license_files() -> list[str]:
    patterns = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt")
    found: list[str] = []
    for name in patterns:
        if (ROOT / name).exists():
            found.append(name)
    return found


def _run_preflight() -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}

    pyproject = _load_pyproject()
    project = pyproject.get("project", {})

    required_project_fields = [
        "name",
        "version",
        "description",
        "readme",
        "requires-python",
        "classifiers",
        "urls",
        "dependencies",
        "optional-dependencies",
    ]
    missing_fields = [field for field in required_project_fields if not project.get(field)]
    checks["project_required_fields_present"] = not missing_fields
    if missing_fields:
        errors.append(
            "pyproject.toml missing required project fields for release preflight: "
            + ", ".join(missing_fields)
        )

    license_value = project.get("license")
    checks["project_license_present"] = bool(license_value)
    if not license_value:
        errors.append(
            "pyproject.toml is missing project.license; choose and record an explicit license before PyPI release."
        )

    license_files = _root_license_files()
    checks["root_license_files"] = license_files
    if not license_files:
        errors.append("repository root is missing a LICENSE/COPYING file required for public release")

    install_smoke = _load_json(INSTALL_SMOKE)
    checks["install_smoke_ok"] = bool(install_smoke.get("ok"))
    checks["install_smoke_twine_check_ok"] = bool(install_smoke.get("twine_check_ok"))
    if not checks["install_smoke_ok"]:
        errors.append("install_smoke.json reports ok=false")
    if not checks["install_smoke_twine_check_ok"]:
        errors.append("install_smoke.json reports twine_check_ok=false")

    repo_gate = _load_json(REPO_GATE)
    checks["repo_gate_ok"] = bool(repo_gate.get("ok"))
    if not checks["repo_gate_ok"]:
        errors.append("release_gate_repo_v1.json reports ok=false")

    return {
        "ok": not errors,
        "checks": checks,
        "errors": errors,
        "project_name": project.get("name"),
        "project_version": project.get("version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PyPI release prerequisites.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "reports" / "pypi_release_preflight.json",
        help="Path to write the preflight report JSON.",
    )
    args = parser.parse_args()

    report = _run_preflight()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    label = "PASS" if report["ok"] else "FAIL"
    print(f"pypi_release_preflight: {label}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
