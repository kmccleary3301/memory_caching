from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected YAML mapping")
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return loaded


def _latest_ckpt_meta(profile_dir: Path) -> dict[str, Any] | None:
    metas = sorted(profile_dir.glob("step_*.json"))
    if not metas:
        return None
    return _load_json(metas[-1])


def _row_for_profile(
    *,
    profile: str,
    target_cfg: dict[str, Any],
    checkpoints_root: Path,
    eval_root: Path,
) -> dict[str, Any]:
    target_seq_len = int(target_cfg.get("target_seq_len", 0))
    target_batch = int(target_cfg.get("target_batch_size", 0))
    target_steps = int(target_cfg.get("target_steps", 0))

    meta = _latest_ckpt_meta(checkpoints_root / profile)
    eval_path = eval_root / f"{profile}_periodic_eval.json"
    eval_payload = _load_json(eval_path) if eval_path.exists() else {}

    actual_seq_len = int(meta.get("sequence_length", 0)) if isinstance(meta, dict) else 0
    actual_batch = int(meta.get("batch_size", 0)) if isinstance(meta, dict) else 0
    actual_steps = int(meta.get("global_step", 0)) if isinstance(meta, dict) else 0
    eval_proxy = (
        float(eval_payload.get("proxy_score", 0.0))
        if isinstance(eval_payload, dict) and eval_payload.get("proxy_score") is not None
        else 0.0
    )

    return {
        "profile": profile,
        "target_seq_len": target_seq_len,
        "actual_seq_len": actual_seq_len,
        "target_batch_size": target_batch,
        "actual_batch_size": actual_batch,
        "target_steps": target_steps,
        "actual_steps": actual_steps,
        "sequence_len_match": actual_seq_len == target_seq_len,
        "batch_size_match": actual_batch == target_batch,
        "steps_ratio": float(actual_steps / target_steps) if target_steps > 0 else 0.0,
        "eval_proxy_score": eval_proxy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-yaml", required=True)
    parser.add_argument("--checkpoints-root", default="artifacts/checkpoints")
    parser.add_argument("--eval-root", default="outputs/eval")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    targets = _load_yaml(Path(args.targets_yaml))
    profiles = targets.get("profiles", {})
    if not isinstance(profiles, dict):
        raise SystemExit("targets.profiles must be a mapping")

    rows: list[dict[str, Any]] = []
    for profile in sorted(profiles):
        cfg = profiles[profile]
        if not isinstance(cfg, dict):
            continue
        rows.append(
            _row_for_profile(
                profile=profile,
                target_cfg=cfg,
                checkpoints_root=Path(args.checkpoints_root),
                eval_root=Path(args.eval_root),
            )
        )

    payload = {
        "schema_version": "v1",
        "targets_yaml": args.targets_yaml,
        "checkpoints_root": args.checkpoints_root,
        "eval_root": args.eval_root,
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = ["# Training Parity Table", ""]
    lines.append(f"targets_yaml: {args.targets_yaml}")
    lines.append("")
    lines.append("| Profile | Target Seq | Actual Seq | Target Batch | Actual Batch | Target Steps | Actual Steps | Steps Ratio | Eval Proxy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['target_seq_len']} | {row['actual_seq_len']} | {row['target_batch_size']} | {row['actual_batch_size']} | {row['target_steps']} | {row['actual_steps']} | {row['steps_ratio']:.4f} | {row['eval_proxy_score']:.6f} |"
        )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
