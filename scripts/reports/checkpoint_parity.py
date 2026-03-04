from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"checkpoint must be a dict: {path}")
    return payload


def _max_state_diff(a: dict[str, Any], b: dict[str, Any]) -> float:
    state_a = a.get("model_state", {})
    state_b = b.get("model_state", {})
    if not isinstance(state_a, dict) or not isinstance(state_b, dict):
        raise SystemExit("both checkpoints must include model_state dictionaries")
    if set(state_a.keys()) != set(state_b.keys()):
        raise SystemExit("model_state keys do not match")

    max_diff = 0.0
    for key in state_a:
        ta = state_a[key]
        tb = state_b[key]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            raise SystemExit(f"model_state[{key}] must be tensor in both checkpoints")
        diff = (ta - tb).abs().max().item()
        if diff > max_diff:
            max_diff = float(diff)
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True, help="full-run checkpoint")
    parser.add_argument("--actual", required=True, help="resume-run checkpoint")
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    expected_path = Path(args.expected)
    actual_path = Path(args.actual)
    expected = _load_checkpoint(expected_path)
    actual = _load_checkpoint(actual_path)

    expected_step = int(expected.get("global_step", -1))
    actual_step = int(actual.get("global_step", -1))
    max_diff = _max_state_diff(expected, actual)
    ok = expected_step == actual_step and max_diff <= args.tol

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected": str(expected_path),
        "actual": str(actual_path),
        "expected_step": expected_step,
        "actual_step": actual_step,
        "max_abs_param_diff": max_diff,
        "tolerance": args.tol,
        "ok": ok,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if not ok:
        raise SystemExit(
            f"checkpoint parity failed: expected_step={expected_step}, actual_step={actual_step}, max_diff={max_diff}"
        )
    print("checkpoint parity: PASS")


if __name__ == "__main__":
    main()
