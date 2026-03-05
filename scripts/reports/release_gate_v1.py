from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_BENCH_RUNS = {"niah", "mqar", "longbench", "retrieval"}


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected JSON object")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default="outputs/reports/release_gate_v1.json")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    required_files = [
        root / "outputs/checks/phase2_summary.json",
        root / "outputs/checks/phase3_summary.json",
        root / "outputs/checks/phase4_summary.json",
        root / "outputs/checks/resume_consistency.json",
        root / "outputs/reports/phase3_benchmark_trend.json",
        root / "outputs/reports/phase3_parity_dashboard.json",
        root / "outputs/reports/phase3_stat_summary.json",
        root / "outputs/reports/phase3_artifact_checksums.json",
        root / "outputs/reports/training_parity_table.json",
        root / "configs/bench/smoke_targets.yaml",
        root / "docs/CLAIM_TO_EVIDENCE_MATRIX.md",
        root / "docs/reproduction_report.md",
        root / "docs/PROGRESS_LEDGER.md",
    ]

    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    for path in required_files:
        ok = path.exists()
        checks.append({"name": f"exists:{path}", "ok": ok})
        if not ok:
            errors.append(f"missing required file: {path}")

    legacy_target_path = root / "configs/bench/paper_targets.yaml"
    legacy_target_absent = not legacy_target_path.exists()
    checks.append(
        {
            "name": "absent:configs/bench/paper_targets.yaml",
            "ok": legacy_target_absent,
        }
    )
    if not legacy_target_absent:
        errors.append(
            "legacy benchmark target config configs/bench/paper_targets.yaml must be removed"
        )

    independent_manifests = sorted(
        (root / "outputs/independent_repro").glob("*/manifest.json")
    )
    independent_ok = len(independent_manifests) > 0
    checks.append(
        {
            "name": "exists_any:outputs/independent_repro/*/manifest.json",
            "ok": independent_ok,
        }
    )
    if not independent_ok:
        errors.append("missing independent repro manifest under outputs/independent_repro/")

    phase_summaries = {}
    for phase in ("phase2", "phase3", "phase4"):
        summary_path = root / f"outputs/checks/{phase}_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        phase_summaries[phase] = summary
        if not bool(summary.get("ok", False)):
            errors.append(f"{summary_path}: summary ok=false")

    resume_path = root / "outputs/checks/resume_consistency.json"
    if resume_path.exists():
        resume_summary = _load_json(resume_path)
        if not bool(resume_summary.get("ok", False)):
            errors.append(f"{resume_path}: resume consistency check failed")

    trend_path = root / "outputs/reports/phase3_benchmark_trend.json"
    if trend_path.exists():
        trend = _load_json(trend_path)
        latest = trend.get("latest_by_run_type", {})
        if not isinstance(latest, dict):
            errors.append(f"{trend_path}: latest_by_run_type must be an object")
        else:
            seen = {str(k).strip().lower() for k in latest.keys()}
            missing_runs = sorted(EXPECTED_BENCH_RUNS - seen)
            if missing_runs:
                errors.append(
                    f"{trend_path}: missing benchmark run types in trend report: {missing_runs}"
                )

    parity_path = root / "outputs/reports/phase3_parity_dashboard.json"
    if parity_path.exists():
        parity = _load_json(parity_path)
        rows = parity.get("rows", [])
        if not isinstance(rows, list) or len(rows) == 0:
            errors.append(f"{parity_path}: rows must be a non-empty list")
        else:
            below = [
                row for row in rows
                if isinstance(row, dict) and str(row.get("status", "")) == "below_target"
            ]
            if below:
                errors.append(f"{parity_path}: contains below_target rows")

    stats_path = root / "outputs/reports/phase3_stat_summary.json"
    if stats_path.exists():
        stats = _load_json(stats_path)
        rows = stats.get("rows", [])
        if not isinstance(rows, list) or len(rows) == 0:
            errors.append(f"{stats_path}: rows must be a non-empty list")

    checksum_path = root / "outputs/reports/phase3_artifact_checksums.json"
    if checksum_path.exists():
        checksum = _load_json(checksum_path)
        if int(checksum.get("file_count", 0)) <= 0:
            errors.append(f"{checksum_path}: file_count must be positive")

    training_parity_path = root / "outputs/reports/training_parity_table.json"
    if training_parity_path.exists():
        training_parity = _load_json(training_parity_path)
        rows = training_parity.get("rows", [])
        if not isinstance(rows, list) or len(rows) == 0:
            errors.append(f"{training_parity_path}: rows must be a non-empty list")

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "ok": len(errors) == 0,
        "errors": errors,
        "checks": checks,
        "phase_summaries": phase_summaries,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n")

    if errors:
        raise SystemExit("\n".join(errors))

    print("release gate v1: PASS")


if __name__ == "__main__":
    main()
