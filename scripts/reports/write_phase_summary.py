from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    try:
        loaded = json.loads(path.read_text())
        if isinstance(loaded, (dict, list)):
            return loaded, None
        return None, "json_root_must_be_object_or_array"
    except Exception as exc:
        return None, str(exc)


def _extract_numeric_fields(obj: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in obj.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input", action="append", dest="inputs", default=[])
    args = parser.parse_args()

    if len(args.inputs) == 0:
        raise SystemExit("at least one --input is required")

    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    parse_errors: list[str] = []

    for raw_path in args.inputs:
        path = Path(raw_path)
        entry: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            missing.append(str(path))
            entries.append(entry)
            continue

        payload, err = _load_json(path)
        if err is not None:
            entry["json_ok"] = False
            entry["error"] = err
            parse_errors.append(f"{path}: {err}")
            entries.append(entry)
            continue

        entry["json_ok"] = True
        if isinstance(payload, dict):
            entry["keys"] = sorted(payload.keys())
            entry["numeric_fields"] = _extract_numeric_fields(payload)
        else:
            entry["size"] = len(payload)
        entries.append(entry)

    summary = {
        "phase": args.phase,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": len(missing) == 0 and len(parse_errors) == 0,
        "input_count": len(args.inputs),
        "missing_inputs": missing,
        "parse_errors": parse_errors,
        "entries": entries,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote phase summary: {out}")


if __name__ == "__main__":
    main()
