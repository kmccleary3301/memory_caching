from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunnerName = Literal["niah", "mqar", "longbench", "retrieval"]


@dataclass(frozen=True)
class BenchmarkConfig:
    runner: RunnerName
    task: str
    lengths: tuple[int, ...]
    seed: int
    adapter: str

    def __post_init__(self) -> None:
        if len(self.lengths) == 0:
            raise ValueError("lengths must be non-empty")
        if any(x <= 0 for x in self.lengths):
            raise ValueError("all lengths must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.task:
            raise ValueError("task must be non-empty")
        if not self.adapter:
            raise ValueError("adapter must be non-empty")
