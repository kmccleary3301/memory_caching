from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _is_legacy_manifest(path: Path) -> tuple[bool, str]:
    try:
        loaded = json.loads(path.read_text())
    except Exception as exc:
        return True, f"invalid_json:{exc}"

    if not isinstance(loaded, dict):
        return True, "json_root_not_object"

    schema_version = str(loaded.get("schema_version", ""))
    if schema_version != "v1":
        return True, f"schema_version={schema_version or 'missing'}"

    required = {
        "run_type",
        "utc_timestamp",
        "runner_version",
        "dataset_revision",
        "metrics_file",
        "rows_file",
        "summary_csv_file",
        "report_file",
    }
    missing = sorted(key for key in required if key not in loaded)
    if missing:
        return True, f"missing_keys={missing}"

    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/benchmarks")
    parser.add_argument("--quarantine-root", default="outputs/quarantine")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--report",
        default="outputs/reports/quarantine_legacy_artifacts.json",
    )
    args = parser.parse_args()

    root = Path(args.root)
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        print(f"no manifest.json found under {root}")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine_root = Path(args.quarantine_root) / stamp
    quarantined: list[dict[str, Any]] = []

    for manifest in manifests:
        legacy, reason = _is_legacy_manifest(manifest)
        if not legacy:
            continue

        source_dir = manifest.parent
        target_dir = quarantine_root / source_dir.name
        row = {
            "manifest": str(manifest),
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "reason": reason,
            "moved": bool(args.apply),
        }
        quarantined.append(row)

        if args.apply:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            if source_dir.exists():
                shutil.move(str(source_dir), str(target_dir))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "apply": bool(args.apply),
        "quarantine_root": str(quarantine_root),
        "quarantined_count": len(quarantined),
        "quarantined": quarantined,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"quarantine scan complete: {len(quarantined)} candidates")


if __name__ == "__main__":
    main()
