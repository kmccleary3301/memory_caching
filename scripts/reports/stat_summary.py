from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_metrics(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return loaded


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return float(math.sqrt(var))


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(1.96 * _std(values) / math.sqrt(len(values)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="benchmark output root")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    metrics_files = sorted(root.rglob("metrics.json"))
    if not metrics_files:
        raise SystemExit(f"no metrics.json files found under {root}")

    grouped: dict[str, list[float]] = {}
    for metrics_path in metrics_files:
        metrics = _load_metrics(metrics_path)
        run_type = str(metrics.get("benchmark", "unknown")).strip().lower()
        value = float(metrics.get("mean_accuracy", 0.0))
        grouped.setdefault(run_type, []).append(value)

    rows: list[dict[str, Any]] = []
    for run_type in sorted(grouped):
        vals = grouped[run_type]
        rows.append(
            {
                "benchmark": run_type,
                "runs": len(vals),
                "mean": _mean(vals),
                "std": _std(vals),
                "ci95_half_width": _ci95(vals),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
        )

    payload = {
        "schema_version": "v1",
        "root": str(root),
        "metrics_file_count": len(metrics_files),
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = ["# Statistical Summary", ""]
    lines.append(f"root: {root}")
    lines.append(f"metrics_file_count: {len(metrics_files)}")
    lines.append("")
    lines.append("| Benchmark | Runs | Mean | Std | CI95 | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['runs']} | {row['mean']:.6f} | {row['std']:.6f} | {row['ci95_half_width']:.6f} | {row['min']:.6f} | {row['max']:.6f} |"
        )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
