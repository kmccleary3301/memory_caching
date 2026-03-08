from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class FenwickBucketState:
    summary: Tensor
    size: int
    start: int
    end: int


@dataclass(frozen=True)
class SingleBatchLogLinearState:
    buckets: tuple[FenwickBucketState, ...]
    position: int


@dataclass(frozen=True)
class LogLinearState:
    batch_states: tuple[SingleBatchLogLinearState, ...]
