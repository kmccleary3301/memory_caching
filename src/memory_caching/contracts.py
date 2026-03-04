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
    """
    Backend contract for Memory Caching.

    Tensor shapes:
    - `k_t`, `v_t`, `q_t`: `[batch_size, num_heads, head_dim]`
    - `apply(...) -> Tensor`: `[batch_size, num_heads, head_dim]`
    """

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


def ensure_backend_state(state: Any, *, stage: str) -> Any:
    if state is None:
        raise RuntimeError(f"backend returned None state at stage={stage}")
    return state


def ensure_head_tensor(
    name: str,
    value: Tensor,
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")

    expected = (batch_size, num_heads, head_dim)
    if tuple(value.shape) != expected:
        raise RuntimeError(f"{name} expected shape {expected}, got {tuple(value.shape)}")

    if device is not None and value.device != device:
        raise RuntimeError(f"{name} expected device {device}, got {value.device}")

    if dtype is not None and value.dtype != dtype:
        raise RuntimeError(f"{name} expected dtype {dtype}, got {value.dtype}")

    return value
