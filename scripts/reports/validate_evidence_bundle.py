from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
VALID_RUN_TYPES = {"niah", "mqar", "longbench", "retrieval"}


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: json root must be an object")
    return loaded


def _validate_manifest(path: Path) -> list[str]:
    try:
        data = _load_json(path)
    except Exception as exc:
        return [f"{path}: failed to parse manifest: {exc}"]
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in data]
    errs: list[str] = []
    if missing:
        errs.append(f"{path}: missing keys {missing}")

    if data.get("schema_version") != "v1":
        errs.append(f"{path}: unsupported schema_version={data.get('schema_version')}")

    run_type = str(data.get("run_type", "")).strip().lower()
    if run_type not in VALID_RUN_TYPES:
        errs.append(f"{path}: invalid run_type={run_type}")

    for key in ["metrics_file", "rows_file", "summary_csv_file", "report_file"]:
        fp = Path(data.get(key, ""))
        if not fp.exists():
            errs.append(f"{path}: referenced file does not exist: {fp}")

    metrics_file = Path(data.get("metrics_file", ""))
    if metrics_file.exists():
        try:
            metrics = _load_json(metrics_file)
        except Exception as exc:
            errs.append(f"{path}: failed to parse metrics_file={metrics_file}: {exc}")
            return errs

        if "mean_accuracy" not in metrics:
            errs.append(f"{metrics_file}: missing mean_accuracy")
        else:
            try:
                float(metrics["mean_accuracy"])
            except Exception:
                errs.append(f"{metrics_file}: mean_accuracy must be numeric")

        rows = metrics.get("rows")
        if not isinstance(rows, list):
            errs.append(f"{metrics_file}: rows must be a list")

        benchmark = str(metrics.get("benchmark", "")).strip().lower()
        if benchmark and benchmark != run_type:
            errs.append(
                f"{metrics_file}: benchmark={benchmark} does not match run_type={run_type}"
            )

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
