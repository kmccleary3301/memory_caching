from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any


def write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n")
        return
    keys = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report_md(path: Path, benchmark: str, rows: list[dict[str, Any]], mean_accuracy: float) -> None:
    lines = [f"# {benchmark} report", "", f"mean_accuracy: {mean_accuracy:.4f}", "", "rows:"]
    for row in rows:
        lines.append(f"- {row}")
    path.write_text("\n".join(lines) + "\n")
