from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _extract_generated_at_utc(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("generated_at_utc:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    raise SystemExit("docs/PAPER_TO_CODE.md is missing generated_at_utc header")


def _load_map_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected mapping")
    return loaded


def _validate_mapped_paths(root: Path, data: dict[str, Any]) -> None:
    sections = data.get("sections", [])
    if not isinstance(sections, list):
        raise SystemExit("map yaml: sections must be a list")
    errors: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        items = section.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            mechanism = str(item.get("mechanism", "unknown"))
            code_paths = item.get("code", [])
            if not isinstance(code_paths, list):
                continue
            for rel in code_paths:
                path = root / str(rel)
                if not path.exists():
                    errors.append(
                        f"map path missing for mechanism '{mechanism}': {rel}"
                    )
    if errors:
        raise SystemExit("\n".join(errors))


def _validate_schema(data: dict[str, Any]) -> None:
    sections = data.get("sections", [])
    if not isinstance(sections, list):
        raise SystemExit("map yaml: sections must be a list")

    errors: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_name = str(section.get("name", "unknown"))
        items = section.get("items", [])
        if not isinstance(items, list):
            errors.append(f"section '{section_name}' items must be a list")
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            mechanism = str(item.get("mechanism", "unknown"))
            symbols = item.get("symbols", [])
            reason = str(item.get("symbol_coverage_reason", "")).strip()

            has_symbols = isinstance(symbols, list) and len([s for s in symbols if str(s).strip()]) > 0
            has_reason = len(reason) > 0

            if not has_symbols and not has_reason:
                errors.append(
                    f"mechanism '{mechanism}' must include either non-empty symbols or symbol_coverage_reason"
                )
            if has_symbols and has_reason:
                errors.append(
                    f"mechanism '{mechanism}' should not define both symbols and symbol_coverage_reason"
                )
    if errors:
        raise SystemExit("\n".join(errors))


def _validate_optional_symbols(root: Path, data: dict[str, Any]) -> None:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    sections = data.get("sections", [])
    if not isinstance(sections, list):
        return
    errors: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        items = section.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            mechanism = str(item.get("mechanism", "unknown"))
            symbols = item.get("symbols", [])
            if not isinstance(symbols, list):
                continue
            for spec in symbols:
                text = str(spec).strip()
                if not text:
                    continue
                if "::" not in text:
                    errors.append(
                        f"invalid symbol spec for mechanism '{mechanism}': {text} (expected module::symbol)"
                    )
                    continue
                module_name, symbol_name = text.split("::", 1)
                try:
                    module = importlib.import_module(module_name)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"failed to import module '{module_name}' for mechanism '{mechanism}': {exc}"
                    )
                    continue
                if not hasattr(module, symbol_name):
                    errors.append(
                        f"missing symbol '{symbol_name}' in module '{module_name}' for mechanism '{mechanism}'"
                    )
    if errors:
        raise SystemExit("\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="validate map schema/paths/symbols without markdown sync comparison",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    map_yaml = root / "docs/paper_to_code_map.yaml"
    paper_to_code_md = root / "docs/PAPER_TO_CODE.md"
    generator = root / "scripts/reports/generate_paper_to_code.py"

    if not map_yaml.exists():
        raise SystemExit(f"missing mapping source: {map_yaml}")
    if not paper_to_code_md.exists():
        raise SystemExit(f"missing generated mapping doc: {paper_to_code_md}")
    if not generator.exists():
        raise SystemExit(f"missing generator script: {generator}")

    current_md = paper_to_code_md.read_text()
    map_data = _load_map_yaml(map_yaml)
    _validate_schema(map_data)
    _validate_mapped_paths(root, map_data)
    _validate_optional_symbols(root, map_data)
    if args.schema_only:
        print("paper_to_code_schema: PASS")
        return
    generated_at = _extract_generated_at_utc(current_md)

    with tempfile.TemporaryDirectory(prefix="paper_to_code_sync_") as tmp_dir:
        tmp_out = Path(tmp_dir) / "PAPER_TO_CODE.expected.md"
        subprocess.run(
            [
                sys.executable,
                "scripts/reports/generate_paper_to_code.py",
                "--map-yaml",
                "docs/paper_to_code_map.yaml",
                "--out-md",
                str(tmp_out),
                "--generated-at-utc",
                generated_at,
            ],
            check=True,
            cwd=str(root),
        )
        expected_md = tmp_out.read_text()

    if current_md != expected_md:
        raise SystemExit(
            "docs/PAPER_TO_CODE.md is out of sync with docs/paper_to_code_map.yaml. "
            "Run: uv run python scripts/reports/generate_paper_to_code.py"
        )

    print("paper_to_code_sync: PASS")


if __name__ == "__main__":
    main()
