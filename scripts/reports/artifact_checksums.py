from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _collect_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(p for p in sorted(path.rglob("*")) if p.is_file())
        elif path.is_file():
            out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        default=[],
        help="file or directory path (repeatable)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if len(args.paths) == 0:
        raise SystemExit("at least one --path must be provided")

    files = _collect_files(args.paths)
    if len(files) == 0:
        raise SystemExit("no files found for checksum calculation")

    rows: list[dict[str, Any]] = []
    for path in files:
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    payload = {
        "schema_version": "v1",
        "file_count": len(rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote checksums for {len(rows)} files")


if __name__ == "__main__":
    main()
