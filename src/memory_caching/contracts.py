from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class SegmentCache:
    mem_state: Any
    seg_context: Tensor
    seg_len: int


@runtime_checkable
class MemoryBackend(Protocol):
    def init_state(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any: ...

    def update(self, state: Any, k_t: Tensor, v_t: Tensor) -> Any: ...

    def apply(self, state: Any, q_t: Tensor) -> Tensor: ...


@runtime_checkable
class MixableMemoryBackend(MemoryBackend, Protocol):
    def mix_states(self, states: Sequence[Any], weights: Tensor) -> Any: ...
