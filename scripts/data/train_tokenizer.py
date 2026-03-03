from __future__ import annotations

import argparse
from pathlib import Path
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("tokenizer_model_placeholder\n")

    manifest = {
        "stage": "tokenizer_train",
        "config": args.config,
        "output": str(output),
    }
    Path("artifacts/tokenizer_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
