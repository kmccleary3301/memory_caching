from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _extract_generated_at_utc(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("generated_at_utc:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    raise SystemExit("docs/PAPER_TO_CODE.md is missing generated_at_utc header")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
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
