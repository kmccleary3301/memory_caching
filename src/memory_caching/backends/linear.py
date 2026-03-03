from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor


class LinearMemoryBackend(nn.Module):
    """
    Matrix-valued linear memory:

    - update: M_t = M_{t-1} + v_t k_t^T
    - apply:  y_t = M_t q_t
    """

    def __init__(self) -> None:
        super().__init__()

    def init_state(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=dtype,
        )

    def update(self, state: Tensor, k_t: Tensor, v_t: Tensor) -> Tensor:
        outer = torch.einsum("bhd,bhe->bhde", v_t, k_t)
        return state + outer

    def apply(self, state: Tensor, q_t: Tensor) -> Tensor:
        return torch.einsum("bhde,bhe->bhd", state, q_t)

    def mix_states(self, states: Sequence[Tensor], weights: Tensor) -> Tensor:
        if len(states) == 0:
            raise ValueError("states must be non-empty")
        stacked = torch.stack(list(states), dim=2)  # [B,H,S,Dh,Dh]
        return torch.einsum("bhs,bhsde->bhde", weights, stacked)
