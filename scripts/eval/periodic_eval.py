from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _load_checkpoint_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runner", default="niah")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    ckpt_payload = _load_checkpoint_payload(ckpt) if ckpt.exists() else None
    loss_tail: list[float] = []
    global_step = 0
    if ckpt_payload is not None:
        value = ckpt_payload.get("loss_tail", [])
        if isinstance(value, list):
            loss_tail = [float(x) for x in value if isinstance(x, (int, float))]
        global_step = int(ckpt_payload.get("global_step", 0))

    mean_tail_loss = float(sum(loss_tail) / len(loss_tail)) if loss_tail else None
    proxy_score = float(math.exp(-mean_tail_loss)) if mean_tail_loss is not None else None
    payload = {
        "stage": "periodic_eval",
        "runner": args.runner,
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "status": (
            "ok"
            if ckpt.exists() and ckpt_payload is not None
            else ("invalid_checkpoint" if ckpt.exists() else "missing_checkpoint")
        ),
        "global_step": global_step,
        "mean_train_loss_tail": mean_tail_loss,
        "proxy_score": proxy_score,
        "loss_tail_size": len(loss_tail),
    }

    if args.out_json is not None:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"periodic_eval_hook checkpoint={args.checkpoint} runner={args.runner} status={payload['status']}"
    )


if __name__ == "__main__":
    main()
