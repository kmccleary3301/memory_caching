from __future__ import annotations

import argparse
from pathlib import Path
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shard_00000.jsonl").write_text('{"text": "placeholder"}\n')

    manifest = {
        "stage": "data_process",
        "config": args.config,
        "tokenizer": args.tokenizer,
        "output_dir": str(out),
    }
    Path("artifacts/data_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
