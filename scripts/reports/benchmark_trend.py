from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_RUN_TYPES = {"niah", "mqar", "longbench", "retrieval"}


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected JSON object")
    return loaded


def _load_entry(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    run_type = str(manifest.get("run_type", "")).strip().lower()
    if run_type not in VALID_RUN_TYPES:
        raise ValueError(f"{manifest_path}: invalid run_type={run_type}")

    metrics_path = Path(str(manifest.get("metrics_file", "")))
    if not metrics_path.exists():
        raise ValueError(f"{manifest_path}: missing metrics_file={metrics_path}")

    metrics = _read_json(metrics_path)
    mean_accuracy = float(metrics.get("mean_accuracy", 0.0))
    adapter_type = str(
        metrics.get("adapter_type", manifest.get("adapter_type", manifest.get("config", {}).get("adapter_type", "unknown")))
    )
    model_info = manifest.get("model_info", manifest.get("config", {}).get("model_info"))

    return {
        "run_type": run_type,
        "utc_timestamp": str(manifest.get("utc_timestamp", "")),
        "artifact_dir": str(manifest_path.parent),
        "manifest": str(manifest_path),
        "runner_version": str(manifest.get("runner_version", "unknown")),
        "dataset_revision": str(manifest.get("dataset_revision", "unknown")),
        "mean_accuracy": mean_accuracy,
        "adapter_type": adapter_type,
        "model_info": model_info,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Benchmark Trend", ""]
    lines.append(f"generated_at_utc: {payload['generated_at_utc']}")
    lines.append(f"root: {payload['root']}")
    lines.append(f"total_runs: {payload['total_runs']}")
    lines.append("")
    lines.append("## Latest by run type")
    for run_type in sorted(payload["latest_by_run_type"]):
        row = payload["latest_by_run_type"][run_type]
        lines.append(
            f"- {run_type}: mean_accuracy={row['mean_accuracy']:.6f}, utc_timestamp={row['utc_timestamp']}, dataset_revision={row['dataset_revision']}"
        )
    lines.append("")
    lines.append("## History sizes")
    for run_type in sorted(payload["history_by_run_type"]):
        lines.append(f"- {run_type}: {len(payload['history_by_run_type'][run_type])}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise SystemExit(f"no manifest.json found under {root}")

    history: dict[str, list[dict[str, Any]]] = {k: [] for k in VALID_RUN_TYPES}
    for manifest_path in manifests:
        entry = _load_entry(manifest_path)
        history[entry["run_type"]].append(entry)

    for run_type in history:
        history[run_type].sort(key=lambda row: row["utc_timestamp"])

    latest = {
        run_type: rows[-1]
        for run_type, rows in history.items()
        if len(rows) > 0
    }

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "total_runs": sum(len(v) for v in history.values()),
        "history_by_run_type": history,
        "latest_by_run_type": latest,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(Path(args.out_md), out_payload)
    print(f"wrote trend reports: {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
