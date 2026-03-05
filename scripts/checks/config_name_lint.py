from __future__ import annotations

import argparse
from pathlib import Path


def _is_reference_only_yaml(path: Path) -> bool:
    text = path.read_text()
    for line in text.splitlines():
        if line.strip().lower() == "reference_only: true":
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    configs_dir = root / "configs"
    errors: list[str] = []

    if not configs_dir.exists():
        raise SystemExit(f"missing configs directory: {configs_dir}")

    for path in sorted(configs_dir.rglob("paper_*.yaml")):
        if not _is_reference_only_yaml(path):
            rel = path.relative_to(root)
            errors.append(
                f"disallowed config name without reference-only marker: {rel} "
                "(set `reference_only: true` or rename to non-paper_* name)"
            )

    if errors:
        raise SystemExit("\n".join(errors))

    print("config_name_lint: PASS")


if __name__ == "__main__":
    main()
