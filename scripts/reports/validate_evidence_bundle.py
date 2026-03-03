from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "run_type",
    "utc_timestamp",
    "git_commit",
    "runner_version",
    "dataset_revision",
    "config",
    "metrics_file",
    "rows_file",
    "summary_csv_file",
    "report_file",
}


def _validate_manifest(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in data]
    errs: list[str] = []
    if missing:
        errs.append(f"{path}: missing keys {missing}")
    for key in ["metrics_file", "rows_file", "summary_csv_file", "report_file"]:
        fp = Path(data.get(key, ""))
        if not fp.exists():
            errs.append(f"{path}: referenced file does not exist: {fp}")
    return errs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise SystemExit("no manifest.json found")

    errors: list[str] = []
    for mf in manifests:
        errors.extend(_validate_manifest(mf))

    if errors:
        raise SystemExit("\n".join(errors))

    print(f"validated {len(manifests)} evidence bundles")


if __name__ == "__main__":
    main()
