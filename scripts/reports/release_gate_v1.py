from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_BENCH_RUNS = {"niah", "mqar", "longbench", "retrieval"}


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected JSON object")
    return loaded


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _validate_install_smoke_contract(
    *, root: Path, install_smoke_path: Path, errors: list[str]
) -> None:
    install_smoke = _load_json(install_smoke_path)
    if not bool(install_smoke.get("ok", False)):
        errors.append(f"{install_smoke_path}: ok=false")
    if not isinstance(install_smoke.get("generated_at_utc", None), str):
        errors.append(f"{install_smoke_path}: generated_at_utc must be a string")
    if install_smoke.get("twine_check_ok") is not True:
        errors.append(f"{install_smoke_path}: twine_check_ok must be true")
    build_artifacts = install_smoke.get("build_artifacts", {})
    if not isinstance(build_artifacts, dict):
        errors.append(f"{install_smoke_path}: build_artifacts must be an object")
    else:
        for key in ("wheel", "sdist"):
            artifact_ref = str(build_artifacts.get(key, ""))
            if not artifact_ref:
                errors.append(f"{install_smoke_path}: build_artifacts missing '{key}'")
                continue
            artifact_path = _resolve_path(root, artifact_ref)
            if not artifact_path.exists():
                errors.append(f"{install_smoke_path}: missing built artifact {artifact_path}")
    runs = install_smoke.get("runs", [])
    if not isinstance(runs, list) or len(runs) != 3:
        errors.append(f"{install_smoke_path}: runs must contain exactly three entries")
        return

    modes = {str(r.get("mode", "")) for r in runs if isinstance(r, dict)}
    if modes != {"wheel", "sdist", "dev"}:
        errors.append(f"{install_smoke_path}: runs modes must be wheel/sdist/dev, got {sorted(modes)}")

    expected_artifact_kind = {
        "wheel": "wheel",
        "sdist": "sdist",
        "dev": "editable_source",
    }
    for run in runs:
        if not isinstance(run, dict):
            errors.append(f"{install_smoke_path}: each run entry must be an object")
            continue
        for key in ("mode", "artifact_kind", "install_cmd", "eval_artifact", "verification_env"):
            if key not in run:
                errors.append(f"{install_smoke_path}: run entry missing key '{key}'")
        mode = str(run.get("mode", ""))
        artifact_kind = str(run.get("artifact_kind", ""))
        expected_kind = expected_artifact_kind.get(mode)
        if expected_kind is not None and artifact_kind != expected_kind:
            errors.append(
                f"{install_smoke_path}: mode={mode} must record artifact_kind={expected_kind}, got {artifact_kind}"
            )
        install_cmd = str(run.get("install_cmd", ""))
        if "--no-deps" in install_cmd:
            errors.append(f"{install_smoke_path}: install_cmd must not use --no-deps ({install_cmd})")
        if str(run.get("verification_env", "")) != "clean_venv":
            errors.append(
                f"{install_smoke_path}: verification_env must be clean_venv, got {run.get('verification_env')}"
            )
        if run.get("py_typed_ok") is not True:
            errors.append(f"{install_smoke_path}: {mode} must record py_typed_ok=true")
        if run.get("import_forward_ok") is not True:
            errors.append(f"{install_smoke_path}: {mode} must record import_forward_ok=true")
        eval_artifact = str(run.get("eval_artifact", ""))
        if not eval_artifact:
            continue
        eval_path = _resolve_path(root, eval_artifact)
        if not eval_path.exists():
            errors.append(f"{install_smoke_path}: missing eval artifact {eval_path}")

def _require_non_smoke_targets(*, root: Path, errors: list[str]) -> None:
    parity_path = root / "outputs/reports/full_dataset_parity_dashboard.json"
    if not parity_path.exists():
        errors.append(f"scientific mode: missing parity dashboard {parity_path}")
        return

    parity = _load_json(parity_path)
    targets_source = str(parity.get("targets_source", ""))
    if targets_source.endswith("smoke_targets.yaml"):
        errors.append(
            f"scientific mode: {parity_path} still points to smoke targets ({targets_source})"
        )


def _require_model_backed_benchmarks(*, root: Path, errors: list[str]) -> None:
    trend_path = root / "outputs/reports/full_dataset_benchmark_trend.json"
    if not trend_path.exists():
        errors.append(f"scientific mode: missing benchmark trend {trend_path}")
        return

    trend = _load_json(trend_path)
    latest = trend.get("latest_by_run_type", {})
    if not isinstance(latest, dict) or len(latest) == 0:
        errors.append(f"scientific mode: {trend_path} latest_by_run_type must be a non-empty object")
        return

    for run_type, row in latest.items():
        if not isinstance(row, dict):
            errors.append(f"scientific mode: malformed latest row for {run_type}")
            continue
        manifest_ref = str(row.get("manifest", ""))
        if not manifest_ref:
            errors.append(f"scientific mode: missing manifest path for {run_type}")
            continue
        manifest_path = _resolve_path(root, manifest_ref)
        if not manifest_path.exists():
            errors.append(f"scientific mode: missing manifest for {run_type}: {manifest_path}")
            continue

        manifest = _load_json(manifest_path)
        metrics_ref = str(manifest.get("metrics_file", ""))
        metrics_path = (
            _resolve_path(root, metrics_ref)
            if metrics_ref
            else manifest_path.parent / "metrics.json"
        )
        if not metrics_path.exists():
            errors.append(f"scientific mode: missing metrics for {run_type}: {metrics_path}")
            continue

        metrics = _load_json(metrics_path)
        adapter_type = str(
            metrics.get(
                "adapter_type",
                manifest.get("config", {}).get("adapter_type", ""),
            )
        ).strip().lower()
        if adapter_type != "model_backed":
            errors.append(
                f"scientific mode: {run_type} adapter_type={adapter_type or 'missing'} is not model_backed"
            )


def _run_scientific_manifest_lint(*, root: Path, errors: list[str]) -> None:
    lint_script = root / "scripts/checks/scientific_manifest_lint.py"
    if not lint_script.exists():
        errors.append(f"scientific mode: missing validator {lint_script}")
        return
    proc = subprocess.run(
        [
            "python",
            str(lint_script),
            "--project-root",
            str(root),
            "--train-manifest",
            "artifacts/train_manifest.json",
            "--benchmark-root",
            "outputs/benchmarks/full_dataset",
            "--trend-json",
            "outputs/reports/full_dataset_benchmark_trend.json",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr if stderr else stdout
        errors.append(f"scientific_manifest_lint failed: {details}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", choices=["repo", "scientific"], default="repo")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    gate_name = (
        "engineering_release_gate_v1"
        if args.mode == "repo"
        else "scientific_release_gate_v1"
    )
    gate_scope = (
        "repository engineering integrity"
        if args.mode == "repo"
        else "scientific claim eligibility"
    )
    required_files = [
        root / "outputs/checks/phase2_summary.json",
        root / "outputs/checks/phase3_summary.json",
        root / "outputs/checks/phase4_summary.json",
        root / "outputs/checks/resume_consistency.json",
        root / "outputs/checks/install_smoke.json",
        root / "outputs/checks/install_smoke_wheel_eval.json",
        root / "outputs/checks/install_smoke_sdist_eval.json",
        root / "outputs/checks/install_smoke_dev_eval.json",
        root / "outputs/reports/phase3_benchmark_trend.json",
        root / "outputs/reports/phase3_parity_dashboard.json",
        root / "outputs/reports/phase3_stat_summary.json",
        root / "outputs/reports/phase3_artifact_checksums.json",
        root / "outputs/reports/training_parity_table.json",
        root / "configs/bench/smoke_targets.yaml",
        root / "docs/CLAIM_TO_EVIDENCE_MATRIX.md",
        root / "docs/reproduction_report.md",
        root / "docs/PROGRESS_LEDGER.md",
        root / "docs/CONTRIBUTING.md",
        root / "docs/CONTRIBUTOR_DRY_RUN.md",
        root / "docs/ARCHITECTURE.md",
        root / "docs/ENV_COMPAT_MATRIX.md",
        root / "docs/paper_to_code_map.yaml",
        root / "docs/PAPER_TO_CODE.md",
    ]

    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    for path in required_files:
        ok = path.exists()
        checks.append({"name": f"exists:{path}", "ok": ok})
        if not ok:
            errors.append(f"missing required file: {path}")

    paper_to_code_check = root / "scripts/checks/paper_to_code_sync.py"
    if paper_to_code_check.exists():
        proc = subprocess.run(
            [
                "python",
                str(paper_to_code_check),
                "--project-root",
                str(root),
            ],
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        checks.append({"name": "paper_to_code_sync", "ok": ok})
        if not ok:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            details = stderr if stderr else stdout
            errors.append(f"paper_to_code_sync failed: {details}")

        proc = subprocess.run(
            [
                "python",
                str(paper_to_code_check),
                "--project-root",
                str(root),
                "--schema-only",
            ],
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        checks.append({"name": "paper_to_code_schema", "ok": ok})
        if not ok:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            details = stderr if stderr else stdout
            errors.append(f"paper_to_code_schema failed: {details}")

    config_name_lint = root / "scripts/checks/config_name_lint.py"
    if config_name_lint.exists():
        proc = subprocess.run(
            [
                "python",
                str(config_name_lint),
                "--project-root",
                str(root),
            ],
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        checks.append({"name": "config_name_lint", "ok": ok})
        if not ok:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            details = stderr if stderr else stdout
            errors.append(f"config_name_lint failed: {details}")

    claim_lint = root / "scripts/checks/claim_evidence_lint.py"
    if claim_lint.exists():
        proc = subprocess.run(
            [
                "python",
                str(claim_lint),
                "--project-root",
                str(root),
            ],
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        checks.append({"name": "claim_evidence_lint", "ok": ok})
        if not ok:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            details = stderr if stderr else stdout
            errors.append(f"claim_evidence_lint failed: {details}")

    install_smoke_path = root / "outputs/checks/install_smoke.json"
    if install_smoke_path.exists():
        _validate_install_smoke_contract(
            root=root,
            install_smoke_path=install_smoke_path,
            errors=errors,
        )

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

    if args.mode == "scientific":
        _run_scientific_manifest_lint(root=root, errors=errors)
        _require_non_smoke_targets(root=root, errors=errors)
        _require_model_backed_benchmarks(root=root, errors=errors)

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "gate_name": gate_name,
        "gate_scope": gate_scope,
        "mode": args.mode,
        "ok": len(errors) == 0,
        "errors": errors,
        "checks": checks,
        "phase_summaries": phase_summaries,
    }

    out = Path(args.out) if args.out is not None else Path(
        "outputs/reports/release_gate_repo_v1.json"
        if args.mode == "repo"
        else "outputs/reports/release_gate_scientific_v1.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n")

    if errors:
        raise SystemExit("\n".join(errors))

    print(f"{gate_name}: PASS")


if __name__ == "__main__":
    main()
