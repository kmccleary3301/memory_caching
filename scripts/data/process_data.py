from __future__ import annotations

import argparse
import json
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SPECIAL_FALLBACK_VOCAB = {"<unk>": 1}


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return loaded


def _load_tokenizer_vocab(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("vocab"), list):
            vocab: list[str] = payload["vocab"]
            return {token: idx for idx, token in enumerate(vocab)}
    except Exception:
        pass
    return dict(SPECIAL_FALLBACK_VOCAB)


def _read_source_rows(path: Path, text_field: str, limit: int) -> list[str]:
    rows: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        text = payload.get(text_field)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if text:
            rows.append(text)
        if len(rows) >= limit:
            break
    return rows


def _synthetic_rows(source_name: str, limit: int) -> list[str]:
    return [
        f"{source_name} synthetic record {idx} memory caching benchmark context window"
        for idx in range(limit)
    ]


def _clean_rows(
    rows: list[str],
    *,
    min_chars: int,
    lowercase: bool,
) -> tuple[list[str], dict[str, int]]:
    cleaned: list[str] = []
    seen: set[str] = set()
    dropped_short = 0
    dropped_dupe = 0

    for row in rows:
        text = " ".join(row.split())
        if lowercase:
            text = text.lower()
        if len(text) < min_chars:
            dropped_short += 1
            continue
        if text in seen:
            dropped_dupe += 1
            continue
        seen.add(text)
        cleaned.append(text)

    return cleaned, {
        "input_rows": len(rows),
        "kept_rows": len(cleaned),
        "dropped_short": dropped_short,
        "dropped_duplicates": dropped_dupe,
    }


def _encode(text: str, vocab: dict[str, int]) -> list[int]:
    unk = vocab.get("<unk>", 1)
    return [vocab.get(token, unk) for token in text.split()]


def _sha256_jsonl_files(paths: list[Path]) -> str:
    dig = hashlib.sha256()
    for path in paths:
        for line in path.read_text().splitlines():
            dig.update(line.encode("utf-8"))
            dig.update(b"\n")
    return dig.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    tokenizer_path = Path(args.tokenizer)
    vocab = _load_tokenizer_vocab(tokenizer_path)
    text_field = str(cfg.get("text_field", "text"))
    shard_size = int(cfg.get("shard_size", 256))
    target_records = int(cfg.get("target_records", 512))
    max_records_per_source = int(cfg.get("max_records_per_source", 512))
    split_train = float(cfg.get("split_train_ratio", 0.95))
    split_val = float(cfg.get("split_val_ratio", 0.04))
    split_test = float(cfg.get("split_test_ratio", 0.01))
    if shard_size <= 0:
        raise SystemExit("shard_size must be positive")
    if target_records <= 0:
        raise SystemExit("target_records must be positive")
    if max_records_per_source <= 0:
        raise SystemExit("max_records_per_source must be positive")
    if split_train <= 0 or split_val < 0 or split_test < 0:
        raise SystemExit("split ratios must be non-negative and train must be > 0")
    split_sum = split_train + split_val + split_test
    if abs(split_sum - 1.0) > 1e-6:
        raise SystemExit("split ratios must sum to 1.0")
    min_chars = int(cfg.get("clean_min_chars", 16))
    lowercase = bool(cfg.get("clean_lowercase", False))
    clean_rules_version = str(cfg.get("clean_rules_version", "v1"))

    seed = int(cfg.get("seed", 0))
    rng = random.Random(seed)
    sources = cfg.get("sources", [])
    if not isinstance(sources, list) or len(sources) == 0:
        raise SystemExit("sources must be a non-empty list")

    source_rows: dict[str, list[str]] = {}
    source_weights: list[float] = []
    source_names: list[str] = []
    source_meta: list[dict[str, Any]] = []

    for src in sources:
        if not isinstance(src, dict):
            raise SystemExit("each source must be a mapping")
        name = str(src.get("name", "")).strip()
        uri = str(src.get("uri", "")).strip()
        weight = float(src.get("weight", 0.0))
        if not name:
            raise SystemExit("source.name must be non-empty")
        if weight <= 0:
            raise SystemExit(f"source.weight must be > 0 for {name}")
        path = Path(uri) if uri else Path(f"data/raw/{name}.jsonl")
        if path.exists():
            rows = _read_source_rows(path, text_field, max_records_per_source)
            mode = "file"
        else:
            rows = _synthetic_rows(name, max_records_per_source)
            mode = "synthetic"
        if len(rows) == 0:
            rows = _synthetic_rows(name, max_records_per_source)
            mode = "synthetic-empty-fallback"
        cleaned_rows, clean_stats = _clean_rows(
            rows,
            min_chars=min_chars,
            lowercase=lowercase,
        )
        source_rows[name] = cleaned_rows if len(cleaned_rows) > 0 else rows
        source_names.append(name)
        source_weights.append(weight)
        source_meta.append(
            {
                "name": name,
                "uri": str(path),
                "mode": mode,
                "rows_loaded": len(rows),
                "rows_after_cleaning": len(cleaned_rows),
                "weight": weight,
                "cleaning": clean_stats,
            }
        )

    records: list[dict[str, Any]] = []
    running_idx: dict[str, int] = {name: 0 for name in source_names}
    for record_id in range(target_records):
        source_name = rng.choices(source_names, weights=source_weights, k=1)[0]
        rows = source_rows[source_name]
        idx = running_idx[source_name] % len(rows)
        running_idx[source_name] += 1
        text = rows[idx]
        token_ids = _encode(text, vocab)
        draw = rng.random()
        if draw < split_train:
            split = "train"
        elif draw < split_train + split_val:
            split = "val"
        else:
            split = "test"
        records.append(
            {
                "record_id": record_id,
                "source": source_name,
                "split": split,
                "text": text,
                "token_ids": token_ids,
                "token_count": len(token_ids),
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_rows = {"train": [], "val": [], "test": []}
    for row in records:
        split_rows[str(row["split"])].append(row)

    train_shards: list[str] = []
    val_shards: list[str] = []
    test_shards: list[str] = []

    for start in range(0, len(split_rows["train"]), shard_size):
        shard_idx = start // shard_size
        shard_path = out / f"shard_{shard_idx:05d}.jsonl"
        chunk = split_rows["train"][start : start + shard_size]
        shard_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in chunk))
        train_shards.append(str(shard_path))

    for start in range(0, len(split_rows["val"]), shard_size):
        shard_idx = start // shard_size
        shard_path = out / f"val_shard_{shard_idx:05d}.jsonl"
        chunk = split_rows["val"][start : start + shard_size]
        shard_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in chunk))
        val_shards.append(str(shard_path))

    for start in range(0, len(split_rows["test"]), shard_size):
        shard_idx = start // shard_size
        shard_path = out / f"test_shard_{shard_idx:05d}.jsonl"
        chunk = split_rows["test"][start : start + shard_size]
        shard_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in chunk))
        test_shards.append(str(shard_path))

    shard_files = train_shards + val_shards + test_shards

    index_path = out / "index.json"
    index_payload = {
        "schema_version": "v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": str(tokenizer_path),
        "seed": seed,
        "clean_rules_version": clean_rules_version,
        "clean_min_chars": min_chars,
        "clean_lowercase": lowercase,
        "target_records": target_records,
        "shard_size": shard_size,
        "shards": shard_files,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "test_shards": test_shards,
        "split_counts": {
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
        },
    }
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

    source_distribution: dict[str, int] = {}
    split_distribution = {"train": 0, "val": 0, "test": 0}
    total_tokens = 0
    for row in records:
        name = str(row["source"])
        source_distribution[name] = source_distribution.get(name, 0) + 1
        split = str(row["split"])
        split_distribution[split] = split_distribution.get(split, 0) + 1
        total_tokens += int(row["token_count"])

    manifest = {
        "stage": "data_process",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "tokenizer": str(tokenizer_path),
        "output_dir": str(out),
        "index_file": str(index_path),
        "seed": seed,
        "target_records": target_records,
        "actual_records": len(records),
        "avg_tokens_per_record": float(total_tokens / max(1, len(records))),
        "source_distribution": source_distribution,
        "split_distribution": split_distribution,
        "source_meta": source_meta,
        "clean_rules_version": clean_rules_version,
        "clean_min_chars": min_chars,
        "clean_lowercase": lowercase,
        "shard_count": len(shard_files),
        "shards": shard_files,
        "fingerprint_sha256": _sha256_jsonl_files([Path(p) for p in shard_files])
        if shard_files
        else "",
    }
    Path("artifacts/data_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
