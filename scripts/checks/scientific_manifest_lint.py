from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memory_caching.scientific_manifest import (
    validate_benchmark_manifest,
    validate_train_manifest,
)


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return loaded


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-manifest", default="artifacts/train_manifest.json")
    parser.add_argument("--benchmark-root", default="outputs/benchmarks/full_dataset")
    parser.add_argument("--trend-json", default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    train_manifest_path = _resolve_path(root, args.train_manifest)
    benchmark_root = _resolve_path(root, args.benchmark_root)
    errors: list[str] = []

    if not train_manifest_path.exists():
        errors.append(f"missing train manifest: {train_manifest_path}")
    else:
        train_manifest = _load_json(train_manifest_path)
        errors.extend(validate_train_manifest(train_manifest))
        checkpoint_path = str(train_manifest.get("checkpoint_path", ""))
        if checkpoint_path:
            resolved_checkpoint = _resolve_path(root, checkpoint_path)
            if not resolved_checkpoint.exists():
                errors.append(f"train manifest checkpoint_path does not exist: {resolved_checkpoint}")
        config_path = str(train_manifest.get("config_path", ""))
        if config_path:
            resolved_config = _resolve_path(root, config_path)
            if not resolved_config.exists():
                errors.append(f"train manifest config_path does not exist: {resolved_config}")

    manifests: list[Path]
    if args.trend_json is not None:
        trend_path = _resolve_path(root, args.trend_json)
        if not trend_path.exists():
            errors.append(f"missing trend json: {trend_path}")
            manifests = []
        else:
            trend = _load_json(trend_path)
            latest = trend.get("latest_by_run_type", {})
            manifests = []
            if isinstance(latest, dict):
                for row in latest.values():
                    if not isinstance(row, dict):
                        continue
                    manifest_ref = str(row.get("manifest", ""))
                    if manifest_ref:
                        manifests.append(_resolve_path(root, manifest_ref))
    else:
        manifests = sorted(benchmark_root.rglob("manifest.json"))
    if not manifests:
        errors.append(f"no benchmark manifests found under {benchmark_root}")
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        metrics_file = str(manifest.get("metrics_file", ""))
        metrics_path = _resolve_path(root, metrics_file) if metrics_file else manifest_path.parent / "metrics.json"
        if not metrics_path.exists():
            errors.append(f"missing benchmark metrics for manifest: {manifest_path}")
            continue
        metrics = _load_json(metrics_path)
        errors.extend(validate_benchmark_manifest(manifest, metrics))

    if errors:
        raise SystemExit("\n".join(errors))

    print("scientific_manifest_lint: PASS")


if __name__ == "__main__":
    main()
