from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected YAML mapping")
    return loaded


def _status(delta: float, tolerance: float) -> str:
    if delta >= 0:
        return "meets_or_exceeds"
    if abs(delta) <= tolerance:
        return "within_tolerance"
    return "below_target"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trend-json",
        required=True,
        help="path to phase3_benchmark_trend.json",
    )
    parser.add_argument(
        "--targets-yaml",
        required=True,
        help="path to benchmark target YAML",
    )
    parser.add_argument("--out-md", required=True, help="path to dashboard markdown")
    parser.add_argument("--out-json", default=None, help="optional JSON output path")
    args = parser.parse_args()

    trend = _load_json(Path(args.trend_json))
    targets = _load_yaml(Path(args.targets_yaml))
    latest = trend.get("latest_by_run_type", {})
    if not isinstance(latest, dict):
        raise SystemExit("trend.latest_by_run_type must be an object")
    benchmark_targets = targets.get("benchmarks", {})
    if not isinstance(benchmark_targets, dict):
        raise SystemExit("targets.benchmarks must be a mapping")

    rows: list[dict[str, Any]] = []
    for run_type, row in benchmark_targets.items():
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric", "mean_accuracy"))
        target = float(row.get("target", 0.0))
        tolerance = float(row.get("tolerance", 0.0))
        latest_row = latest.get(run_type, {})
        actual = float(latest_row.get(metric, 0.0)) if isinstance(latest_row, dict) else 0.0
        delta = actual - target
        rows.append(
            {
                "benchmark": run_type,
                "metric": metric,
                "target": target,
                "actual": actual,
                "delta": delta,
                "tolerance": tolerance,
                "status": _status(delta, tolerance),
            }
        )

    out_payload = {
        "schema_version": "v1",
        "trend_source": str(args.trend_json),
        "targets_source": str(args.targets_yaml),
        "targets_schema_version": str(targets.get("schema_version", "")),
        "targets_purpose": str(targets.get("purpose", "")),
        "targets_provenance": targets.get("provenance", []),
        "rows": rows,
    }

    if args.out_json is not None:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n")

    lines = ["# Parity Dashboard", ""]
    lines.append(f"trend_source: {args.trend_json}")
    lines.append(f"targets_source: {args.targets_yaml}")
    lines.append("")
    lines.append("| Benchmark | Metric | Target | Actual | Delta | Tolerance | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['metric']} | {row['target']:.6f} | {row['actual']:.6f} | {row['delta']:.6f} | {row['tolerance']:.6f} | {row['status']} |"
        )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
