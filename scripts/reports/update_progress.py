from __future__ import annotations

import argparse
from pathlib import Path


def weighted_progress(values: dict[str, float]) -> float:
    weights = {
        "phase_0": 0.10,
        "phase_1": 0.25,
        "phase_2": 0.25,
        "phase_3": 0.20,
        "phase_4": 0.15,
        "phase_5": 0.05,
    }
    return sum(weights[k] * values[k] for k in weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-0", type=float, required=True)
    parser.add_argument("--phase-1", type=float, required=True)
    parser.add_argument("--phase-2", type=float, required=True)
    parser.add_argument("--phase-3", type=float, required=True)
    parser.add_argument("--phase-4", type=float, required=True)
    parser.add_argument("--phase-5", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    vals = {
        "phase_0": args.phase_0 / 100.0,
        "phase_1": args.phase_1 / 100.0,
        "phase_2": args.phase_2 / 100.0,
        "phase_3": args.phase_3 / 100.0,
        "phase_4": args.phase_4 / 100.0,
        "phase_5": args.phase_5 / 100.0,
    }

    overall = weighted_progress(vals) * 100.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"overall_completion: {overall:.2f}%\n")
    print(f"overall_completion={overall:.2f}%")


if __name__ == "__main__":
    main()
