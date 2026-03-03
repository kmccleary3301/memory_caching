from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
from typing import Any


@dataclass(frozen=True)
class ArtifactBundle:
    root_dir: Path
    metrics_path: Path
    manifest_path: Path


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
    )


def write_artifacts(
    *,
    bundle: ArtifactBundle,
    run_type: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    bundle.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    manifest = {
        "schema_version": "v1",
        "run_type": run_type,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": config,
        "metrics_file": str(bundle.metrics_path),
    }
    bundle.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
