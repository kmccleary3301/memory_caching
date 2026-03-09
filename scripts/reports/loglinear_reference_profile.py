from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from memory_caching.loglinear import chunked_loglinear_attention, recurrent_loglinear_attention


def _measure(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile recurrent vs chunked log-linear references.")
    parser.add_argument("--out", type=Path, default=Path("outputs/reports/loglinear_reference_profile.json"))
    args = parser.parse_args()

    torch.manual_seed(0)
    rows: list[dict[str, object]] = []
    for seq_len in (16, 32, 64):
        q = torch.randn(2, seq_len, 2, 8)
        k = torch.randn(2, seq_len, 2, 8)
        v = torch.randn(2, seq_len, 2, 8)
        lambda_levels = torch.rand(2, seq_len, 2, 8)
        recurrent_seconds = _measure(lambda: recurrent_loglinear_attention(q, k, v, lambda_levels))
        for chunk_size in (2, 4, 8):
            chunked_seconds = _measure(
                lambda cs=chunk_size: chunked_loglinear_attention(q, k, v, lambda_levels, chunk_size=cs)
            )
            rows.append(
                {
                    "seq_len": seq_len,
                    "chunk_size": chunk_size,
                    "recurrent_seconds": recurrent_seconds,
                    "chunked_seconds": chunked_seconds,
                }
            )

    payload = {
        "schema_version": 1,
        "purpose": "correctness_first_reference_profile",
        "claim_boundary": "Reference timings only; not optimized-path or paper-performance parity evidence.",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
