from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
from typing import Any

from .results import write_report_md, write_rows_jsonl, write_summary_csv


@dataclass(frozen=True)
class ArtifactBundle:
    root_dir: Path
    metrics_path: Path
    manifest_path: Path
    rows_path: Path
    summary_csv_path: Path
    report_md_path: Path


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def create_bundle(base_dir: str | None) -> ArtifactBundle:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(base_dir or "outputs/benchmarks") / stamp
    root.mkdir(parents=True, exist_ok=True)
    return ArtifactBundle(
        root_dir=root,
        metrics_path=root / "metrics.json",
        manifest_path=root / "manifest.json",
        rows_path=root / "rows.jsonl",
        summary_csv_path=root / "summary.csv",
        report_md_path=root / "report.md",
    )


def write_artifacts(
    *,
    bundle: ArtifactBundle,
    run_type: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    runner_version: str,
    dataset_revision: str,
) -> None:
    rows = metrics.get("rows", [])
    mean_accuracy = float(metrics.get("mean_accuracy", 0.0))

    bundle.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    write_rows_jsonl(bundle.rows_path, rows)
    write_summary_csv(bundle.summary_csv_path, rows)
    write_report_md(bundle.report_md_path, run_type, rows, mean_accuracy)

    manifest = {
        "schema_version": "v1",
        "run_type": run_type,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "runner_version": runner_version,
        "dataset_revision": dataset_revision,
        "config": config,
        "metrics_file": str(bundle.metrics_path),
        "rows_file": str(bundle.rows_path),
        "summary_csv_file": str(bundle.summary_csv_path),
        "report_file": str(bundle.report_md_path),
    }
    bundle.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
