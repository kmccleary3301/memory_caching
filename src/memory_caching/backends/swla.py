from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..config import SWLAConfig

Tensor = torch.Tensor


@dataclass(frozen=True)
class SWLAState:
    """Paper-faithful SWLA(c=2) state."""
    m_prev: Tensor
    prev_outer: Tensor


class SWLABackend:
    """
    SWLA(c=2) backend.

      M_t = alpha * M_{t-1} + beta * (v_{t-1} k_{t-1}^T) + lam * (v_t k_t^T)
      y_t = M_t q_t
    """

    def __init__(self, config: SWLAConfig | None = None) -> None:
        self.config = config or SWLAConfig()

    def init_state(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SWLAState:
        zeros = torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=dtype,
        )
        return SWLAState(m_prev=zeros.clone(), prev_outer=zeros.clone())

    def update(self, state: SWLAState, k_t: Tensor, v_t: Tensor) -> SWLAState:
        outer = torch.einsum("bhd,bhe->bhde", v_t, k_t)
        m_t = (
            self.config.alpha * state.m_prev
            + self.config.beta * state.prev_outer
            + self.config.lam * outer
        )
        return SWLAState(m_prev=m_t, prev_outer=outer)

    def apply(self, state: SWLAState, q_t: Tensor) -> Tensor:
        return torch.einsum("bhde,bhe->bhd", state.m_prev, q_t)

    def mix_states(self, states: Sequence[SWLAState], weights: Tensor) -> SWLAState:
        if len(states) == 0:
            raise ValueError("states must be non-empty")

        stacked_prev = torch.stack([s.m_prev for s in states], dim=2)
        stacked_prev_outer = torch.stack([s.prev_outer for s in states], dim=2)
        mixed_prev = torch.einsum("bhs,bhsde->bhde", weights, stacked_prev)
        mixed_prev_outer = torch.einsum("bhs,bhsde->bhde", weights, stacked_prev_outer)
        return SWLAState(m_prev=mixed_prev, prev_outer=mixed_prev_outer)
