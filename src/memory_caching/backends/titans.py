from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from ..config import TitansConfig

Tensor = torch.Tensor


@dataclass(frozen=True)
class TitansState:
    weights: tuple[Tensor, ...]
    biases: tuple[Tensor, ...]
    s_w: tuple[Tensor, ...] | None = None
    s_b: tuple[Tensor, ...] | None = None


class TitansBackend:
    """
    Titans-style deep memory backend.

    Update sketch:
      grad_t = ∇L(M_{t-1}; k_t, v_t)
      if update_convention == "paper":
          S_t = beta * S_{t-1} - eta * grad_t
      else:  # "gradient_descent"
          S_t = beta * S_{t-1} + eta * grad_t
      M_t = alpha * M_{t-1} - S_t
    """

    def __init__(self, config: TitansConfig) -> None:
        self.config = config

    def _shapes(self, head_dim: int) -> list[tuple[int, int]]:
        depth = self.config.memory_depth
        width = self.config.memory_width
        shapes: list[tuple[int, int]] = []
        in_dim = head_dim
        for layer_idx in range(depth):
            out_dim = head_dim if layer_idx == depth - 1 else width
            shapes.append((out_dim, in_dim))
            in_dim = out_dim
        return shapes

    def init_state(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TitansState:
        weights: list[Tensor] = []
        biases: list[Tensor] = []
        s_w: list[Tensor] = []
        s_b: list[Tensor] = []

        for out_dim, in_dim in self._shapes(head_dim):
            w = torch.zeros(
                batch_size,
                num_heads,
                out_dim,
                in_dim,
                device=device,
                dtype=dtype,
            )
            b = torch.zeros(
                batch_size,
                num_heads,
                out_dim,
                device=device,
                dtype=dtype,
            )
            weights.append(w)
            biases.append(b)
            s_w.append(torch.zeros_like(w))
            s_b.append(torch.zeros_like(b))

        return TitansState(
            weights=tuple(weights),
            biases=tuple(biases),
            s_w=tuple(s_w),
            s_b=tuple(s_b),
        )

    def apply(self, state: TitansState, q_t: Tensor) -> Tensor:
        x = q_t
        for idx, (w, b) in enumerate(zip(state.weights, state.biases, strict=True)):
            x = torch.einsum("bhoi,bhi->bho", w, x) + b
            if idx < len(state.weights) - 1:
                x = F.gelu(x)
        return x

    def _loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.config.objective == "l2":
            # Per-memory-state objective; summing over (B,H) keeps update
            # magnitude invariant to batch/head replication.
            return ((pred - target) ** 2).sum(dim=-1).sum()
        if self.config.objective == "dot":
            # Per-memory-state objective; summing over (B,H) keeps update
            # magnitude invariant to batch/head replication.
            return -(pred * target).sum(dim=-1).sum()
        raise ValueError(f"unsupported titans objective: {self.config.objective}")

    def update(self, state: TitansState, k_t: Tensor, v_t: Tensor) -> TitansState:
        requires_graph = self.config.inner_update_mode == "differentiable"

        with torch.enable_grad():
            if requires_graph:
                w_vars = [
                    w if w.requires_grad else w.requires_grad_(True)
                    for w in state.weights
                ]
                b_vars = [
                    b if b.requires_grad else b.requires_grad_(True)
                    for b in state.biases
                ]
            else:
                w_vars = [w.detach().requires_grad_(True) for w in state.weights]
                b_vars = [b.detach().requires_grad_(True) for b in state.biases]
            tmp_state = TitansState(weights=tuple(w_vars), biases=tuple(b_vars))
            pred = self.apply(tmp_state, k_t)
            loss = self._loss(pred, v_t)
            grads = torch.autograd.grad(
                loss,
                [*w_vars, *b_vars],
                create_graph=requires_graph,
                retain_graph=requires_graph,
                allow_unused=False,
            )

        n = len(w_vars)
        g_w = grads[:n]
        g_b = grads[n:]

        beta = self.config.momentum
        eta = self.config.step_size
        alpha = self.config.retention_alpha

        new_w: list[Tensor] = []
        new_b: list[Tensor] = []
        new_s_w: list[Tensor] = []
        new_s_b: list[Tensor] = []

        has_s = state.s_w is not None and state.s_b is not None
        for i, (w, b, gw, gb) in enumerate(zip(w_vars, b_vars, g_w, g_b, strict=True)):
            prev_sw = state.s_w[i] if has_s else torch.zeros_like(w)
            prev_sb = state.s_b[i] if has_s else torch.zeros_like(b)

            if self.config.update_convention == "paper":
                s_w_t = beta * prev_sw - eta * gw
                s_b_t = beta * prev_sb - eta * gb
            else:
                s_w_t = beta * prev_sw + eta * gw
                s_b_t = beta * prev_sb + eta * gb

            next_w = alpha * w - s_w_t
            next_b = alpha * b - s_b_t

            if self.config.inner_update_mode == "stopgrad":
                s_w_t = s_w_t.detach()
                s_b_t = s_b_t.detach()
                next_w = next_w.detach()
                next_b = next_b.detach()

            new_w.append(next_w)
            new_b.append(next_b)
            new_s_w.append(s_w_t)
            new_s_b.append(s_b_t)

        return TitansState(
            weights=tuple(new_w),
            biases=tuple(new_b),
            s_w=tuple(new_s_w),
            s_b=tuple(new_s_b),
        )

    def mix_states(self, states: Sequence[TitansState], weights: Tensor) -> TitansState:
        if len(states) == 0:
            raise ValueError("states must be non-empty")

        num_layers = len(states[0].weights)
        out_w: list[Tensor] = []
        out_b: list[Tensor] = []
        out_sw: list[Tensor] = []
        out_sb: list[Tensor] = []

        for l in range(num_layers):
            sw = torch.stack([s.weights[l] for s in states], dim=2)
            sb = torch.stack([s.biases[l] for s in states], dim=2)
            mw = torch.einsum("bhs,bhsoi->bhoi", weights, sw)
            mb = torch.einsum("bhs,bhso->bho", weights, sb)
            out_w.append(mw)
            out_b.append(mb)

            if states[0].s_w is not None and states[0].s_b is not None:
                ssw = torch.stack([s.s_w[l] for s in states], dim=2)
                ssb = torch.stack([s.s_b[l] for s in states], dim=2)
                msw = torch.einsum("bhs,bhsoi->bhoi", weights, ssw)
                msb = torch.einsum("bhs,bhso->bho", weights, ssb)
                out_sw.append(msw)
                out_sb.append(msb)

        return TitansState(
            weights=tuple(out_w),
            biases=tuple(out_b),
            s_w=tuple(out_sw) if out_sw else None,
            s_b=tuple(out_sb) if out_sb else None,
        )
