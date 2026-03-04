from __future__ import annotations

import argparse
from collections import Counter
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return loaded


def _synthetic_corpus(source_name: str, count: int) -> list[str]:
    lines: list[str] = []
    for idx in range(count):
        lines.append(
            f"{source_name} synthetic sample {idx} memory caching recurrent segment context retrieval"
        )
    return lines


def _read_jsonl_texts(path: Path, text_field: str, limit: int) -> list[str]:
    rows: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        value = payload.get(text_field)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            rows.append(text)
        if len(rows) >= limit:
            break
    return rows


def _collect_corpus(cfg: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    text_field = str(cfg.get("text_field", "text"))
    max_records = int(cfg.get("max_records_per_source", 256))
    input_files = cfg.get("input_files", [])
    source_stats: list[dict[str, Any]] = []

    if not isinstance(input_files, list):
        raise SystemExit("input_files must be a list")

    corpus: list[str] = []
    if len(input_files) == 0:
        synthetic = _synthetic_corpus("default_source", max_records)
        source_stats.append(
            {
                "source": "default_source",
                "path": None,
                "mode": "synthetic",
                "records": len(synthetic),
            }
        )
        return synthetic, source_stats

    for raw in input_files:
        source_path = Path(str(raw))
        source_name = source_path.stem
        if source_path.exists():
            rows = _read_jsonl_texts(source_path, text_field, max_records)
            mode = "file"
        else:
            rows = _synthetic_corpus(source_name, max_records)
            mode = "synthetic"
        corpus.extend(rows)
        source_stats.append(
            {
                "source": source_name,
                "path": str(source_path),
                "mode": mode,
                "records": len(rows),
            }
        )
    return corpus, source_stats


def _build_vocab(
    *,
    corpus: list[str],
    vocab_size: int,
    min_token_freq: int,
) -> tuple[list[str], dict[str, int]]:
    counter: Counter[str] = Counter()
    for text in corpus:
        counter.update(text.split())

    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    vocab = list(SPECIAL_TOKENS)
    for token, freq in sorted_tokens:
        if freq < min_token_freq:
            continue
        if token in vocab:
            continue
        vocab.append(token)
        if len(vocab) >= vocab_size:
            break

    token_counts = {token: counter[token] for token in vocab if token in counter}
    return vocab, token_counts


def _sha256_lines(lines: list[str]) -> str:
    dig = hashlib.sha256()
    for line in lines:
        dig.update(line.encode("utf-8"))
        dig.update(b"\n")
    return dig.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    vocab_size = int(cfg.get("vocab_size", 32000))
    if vocab_size < len(SPECIAL_TOKENS):
        raise SystemExit(f"vocab_size must be >= {len(SPECIAL_TOKENS)}")
    min_token_freq = int(cfg.get("min_token_freq", 1))
    if min_token_freq <= 0:
        raise SystemExit("min_token_freq must be positive")

    corpus, source_stats = _collect_corpus(cfg)
    vocab, token_counts = _build_vocab(
        corpus=corpus,
        vocab_size=vocab_size,
        min_token_freq=min_token_freq,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "schema_version": "v1",
        "tokenizer_type": "whitespace",
        "special_tokens": SPECIAL_TOKENS,
        "vocab_size": len(vocab),
        "vocab": vocab,
        "unk_token": "<unk>",
        "fingerprint_sha256": _sha256_lines(corpus),
    }
    output.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    vocab_path = output.with_suffix(".vocab.txt")
    vocab_path.write_text("\n".join(vocab) + "\n")

    manifest = {
        "stage": "tokenizer_train",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "output": str(output),
        "vocab_file": str(vocab_path),
        "tokenizer_type": "whitespace",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": len(vocab),
        "min_token_freq": min_token_freq,
        "source_stats": source_stats,
        "corpus_records": len(corpus),
        "fingerprint_sha256": model_payload["fingerprint_sha256"],
        "top_token_counts": sorted(
            [{"token": k, "count": int(v)} for k, v in token_counts.items()],
            key=lambda row: (-row["count"], row["token"]),
        )[:25],
    }
    Path("artifacts/tokenizer_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/tokenizer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
