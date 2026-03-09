from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize log-linear pilot benchmark artifacts.")
    parser.add_argument("--bench-root", type=Path, default=Path("outputs/benchmarks/loglinear_pilots"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/loglinear_pilot_report.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/reports/loglinear_pilot_report.md"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for manifest_path in sorted(args.bench_root.glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        model_info = manifest.get("model_info", {})
        metrics_path = Path(manifest.get("metrics_file", ""))
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metric_name = None
        metric_value = None
        for key, value in metrics.items():
            if key in {"benchmark", "adapter_type", "rows"}:
                continue
            metric_name = key
            metric_value = value
            break
        rows.append(
            {
                "task": manifest.get("run_type"),
                "checkpoint": model_info.get("checkpoint_path"),
                "model_family": model_info.get("model_family"),
                "metric": metric_name,
                "result": metric_value,
                "target_interpretation": "pilot-scale model-backed evidence only",
            }
        )

    payload = {
        "schema_version": 1,
        "purpose": "loglinear_pilot_report",
        "claim_boundary": "Pilot evidence is not paper-scale parity.",
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Log-linear Pilot Report",
        "",
        "Pilot-scale model-backed evidence only. This does not establish paper-scale parity.",
        "",
        "| task | checkpoint | model_family | metric | result | target_interpretation |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['checkpoint']} | {row['model_family']} | {row['metric']} | {row['result']} | {row['target_interpretation']} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
