from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected YAML mapping")
    return loaded


def _render_markdown(data: dict[str, Any], source_path: str) -> str:
    paper = str(data.get("paper", "unknown"))
    generated_by = str(data.get("generated_by", "unknown"))
    sections = data.get("sections", [])
    if not isinstance(sections, list):
        raise SystemExit("sections must be a list")

    lines = [
        "# Paper to Code Mapping",
        "",
        f"paper: {paper}",
        f"map_source: {source_path}",
        f"generated_by: {generated_by}",
        f"generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Section | Mechanism | Paper Anchor | Code Paths | Status |",
        "|---|---|---|---|---|",
    ]

    for section in sections:
        if not isinstance(section, dict):
            continue
        section_name = str(section.get("name", "unnamed"))
        items = section.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            mechanism = str(item.get("mechanism", ""))
            paper_anchor = str(item.get("paper_anchor", ""))
            status = str(item.get("status", ""))
            code_paths = item.get("code", [])
            if not isinstance(code_paths, list):
                code_paths = []
            code_cell = "<br>".join(str(p) for p in code_paths)
            lines.append(
                f"| {section_name} | {mechanism} | {paper_anchor} | {code_cell} | {status} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-yaml",
        default="docs/paper_to_code_map.yaml",
        help="path to paper->code YAML mapping",
    )
    parser.add_argument(
        "--out-md",
        default="docs/PAPER_TO_CODE.md",
        help="output markdown path",
    )
    args = parser.parse_args()

    map_path = Path(args.map_yaml)
    out_path = Path(args.out_md)
    data = _load_yaml(map_path)
    markdown = _render_markdown(data, str(map_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

