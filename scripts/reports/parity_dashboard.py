from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="path to metrics.json")
    parser.add_argument("--out", required=True, help="path to dashboard markdown")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Parity Dashboard", "", f"benchmark: {metrics.get('benchmark', 'unknown')}"]
    lines.append(f"mean_accuracy: {metrics.get('mean_accuracy', 0.0)}")
    lines.append("")
    lines.append("rows:")
    for row in metrics.get("rows", []):
        lines.append(f"- {row}")

    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
