from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FenwickBucket:
    level: int
    start: int
    end: int
    size: int


def _lowbit(n: int) -> int:
    if n <= 0:
        raise ValueError(f"lowbit requires n > 0, received {n}")
    return n & -n


def max_active_levels(seq_len: int) -> int:
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, received {seq_len}")
    return max(1, seq_len.bit_length())


def fenwick_prefix_buckets(prefix_len: int, *, start_level: int = 1) -> list[FenwickBucket]:
    """Return a recent->distant Fenwick decomposition of [0, prefix_len).

    Level 0 is reserved for the self token in the current reference contract.
    """

    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, received {prefix_len}")

    buckets: list[FenwickBucket] = []
    end = prefix_len
    level = start_level
    while end > 0:
        size = _lowbit(end)
        start = end - size
        buckets.append(FenwickBucket(level=level, start=start, end=end, size=size))
        end = start
        level += 1
    return buckets


def timestep_buckets(timestep: int) -> list[FenwickBucket]:
    """Return the bucketization used by the reference implementation at timestep t.

    Contract:
    - level 0 is always the self token [t, t+1)
    - older tokens [0, t) are covered by a recent->distant Fenwick decomposition
    """

    if timestep < 0:
        raise ValueError(f"timestep must be >= 0, received {timestep}")

    return [FenwickBucket(level=0, start=timestep, end=timestep + 1, size=1)] + fenwick_prefix_buckets(
        timestep,
        start_level=1,
    )


def hierarchical_level_index(seq_len: int) -> torch.Tensor:
    """Materialize a dense level-index matrix for the reference contract.

    Returns shape [T, T] with:
    - `-1` for non-causal positions
    - `level >= 0` for causal source/query pairs
    """

    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, received {seq_len}")

    level_index = torch.full((seq_len, seq_len), -1, dtype=torch.long)
    for t in range(seq_len):
        for bucket in timestep_buckets(t):
            level_index[t, bucket.start : bucket.end] = bucket.level
    return level_index
