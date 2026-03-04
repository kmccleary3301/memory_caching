from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SweepStep:
    name: str
    command: tuple[str, ...]


def _default_steps(root: Path) -> list[SweepStep]:
    return [
        SweepStep(
            name="niah",
            command=(
                "uv",
                "run",
                "mc",
                "bench",
                "niah",
                "--adapter",
                "all",
                "--tasks",
                "s_niah_1,s_niah_2,s_niah_3",
                "--context-lengths",
                "4096,8192",
                "--samples-per-length",
                "8",
                "--seed",
                "0",
                "--out-dir",
                str(root / "niah"),
            ),
        ),
        SweepStep(
            name="mqar",
            command=(
                "uv",
                "run",
                "mc",
                "bench",
                "mqar",
                "--adapter",
                "all",
                "--samples",
                "32",
                "--pair-grid",
                "8,16",
                "--query-grid",
                "1,4",
                "--seed",
                "0",
                "--out-dir",
                str(root / "mqar"),
            ),
        ),
        SweepStep(
            name="longbench",
            command=(
                "uv",
                "run",
                "mc",
                "bench",
                "longbench",
                "--adapter",
                "all",
                "--tasks",
                "single_doc_qa,multi_doc_qa,summarization,few_shot,code",
                "--samples-per-task",
                "4",
                "--seed",
                "0",
                "--out-dir",
                str(root / "longbench"),
            ),
        ),
        SweepStep(
            name="retrieval",
            command=(
                "uv",
                "run",
                "mc",
                "bench",
                "retrieval",
                "--adapter",
                "all",
                "--datasets",
                "swde,squad,fda",
                "--truncation-lengths",
                "512,1024,2048,16384",
                "--samples-per-dataset",
                "4",
                "--seed",
                "0",
                "--out-dir",
                str(root / "retrieval"),
            ),
        ),
    ]


def _run_step(
    *,
    step: SweepStep,
    marker_dir: Path,
    retries: int,
    timeout_sec: int,
    force: bool,
) -> None:
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{step.name}.done.json"
    if marker.exists() and not force:
        print(f"skip {step.name}: marker exists ({marker})")
        return

    attempts = max(1, retries + 1)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        started = datetime.now(timezone.utc)
        try:
            proc = subprocess.run(
                step.command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            payload = {
                "step": step.name,
                "started_at_utc": started.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
                "command": list(step.command),
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
            marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"completed {step.name} on attempt {attempt}")
            return
        except Exception as exc:
            last_exc = exc
            print(f"attempt {attempt}/{attempts} failed for {step.name}: {exc}")

    if last_exc is not None:
        raise SystemExit(f"step failed after retries: {step.name}: {last_exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/benchmarks/sweeps/default")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report",
        default="outputs/reports/benchmark_sweep_report.json",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    marker_dir = root / "_markers"
    steps = _default_steps(root)

    for step in steps:
        _run_step(
            step=step,
            marker_dir=marker_dir,
            retries=args.retries,
            timeout_sec=args.timeout_sec,
            force=bool(args.force),
        )

    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "markers": [str(p) for p in sorted(marker_dir.glob("*.done.json"))],
        "step_count": len(steps),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n")
    print(f"sweep complete: {len(steps)} steps")


if __name__ == "__main__":
    main()
