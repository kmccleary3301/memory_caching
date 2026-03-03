from __future__ import annotations

import argparse
from pathlib import Path
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "step_000000.pt").write_text("checkpoint_placeholder\n")

    meta = {
        "stage": "train",
        "config": args.config,
        "data_dir": args.data_dir,
        "checkpoint_dir": str(ckpt_dir),
    }
    Path("artifacts/train_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/train_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
