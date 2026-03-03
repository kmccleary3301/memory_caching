from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from ..config import DLAConfig

Tensor = torch.Tensor


@dataclass(frozen=True)
class DLAState:
    weights: tuple[Tensor, ...]  # each [B,H,Out,In]
    biases: tuple[Tensor, ...]   # each [B,H,Out]
    vel_w: tuple[Tensor, ...] | None = None
    vel_b: tuple[Tensor, ...] | None = None


class DLABackend:
    def __init__(self, config: DLAConfig) -> None:
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
    ) -> DLAState:
        weights: list[Tensor] = []
        biases: list[Tensor] = []
        vel_w: list[Tensor] = []
        vel_b: list[Tensor] = []

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
            vel_w.append(torch.zeros_like(w))
            vel_b.append(torch.zeros_like(b))

        return DLAState(
            weights=tuple(weights),
            biases=tuple(biases),
            vel_w=tuple(vel_w),
            vel_b=tuple(vel_b),
        )

    def apply(self, state: DLAState, q_t: Tensor) -> Tensor:
        x = q_t
        for idx, (w, b) in enumerate(zip(state.weights, state.biases, strict=True)):
            x = torch.einsum("bhoi,bhi->bho", w, x) + b
            if idx < len(state.weights) - 1:
                x = F.gelu(x)
        return x

    def _compute_loss(self, pred: Tensor, v_t: Tensor) -> Tensor:
        if self.config.objective == "dot":
            return -(pred * v_t).sum(dim=-1).mean()
        if self.config.objective == "l2":
            return ((pred - v_t) ** 2).mean()
        raise ValueError(f"unsupported objective: {self.config.objective}")

    def update(self, state: DLAState, k_t: Tensor, v_t: Tensor) -> DLAState:
        requires_graph = self.config.inner_update_mode == "differentiable"

        with torch.enable_grad():
            weight_vars = [w.detach().requires_grad_(True) for w in state.weights]
            bias_vars = [b.detach().requires_grad_(True) for b in state.biases]

            temp_state = DLAState(weights=tuple(weight_vars), biases=tuple(bias_vars))
            pred = self.apply(temp_state, k_t)
            loss = self._compute_loss(pred, v_t)

            grads = torch.autograd.grad(
                loss,
                [*weight_vars, *bias_vars],
                create_graph=requires_graph,
                retain_graph=requires_graph,
                allow_unused=False,
            )

        n = len(weight_vars)
        grad_w = grads[:n]
        grad_b = grads[n:]

        step = self.config.step_size
        momentum = self.config.momentum

        new_weights: list[Tensor] = []
        new_biases: list[Tensor] = []
        new_vel_w: list[Tensor] = []
        new_vel_b: list[Tensor] = []

        has_velocity = state.vel_w is not None and state.vel_b is not None
        for idx, (w, b, gw, gb) in enumerate(
            zip(weight_vars, bias_vars, grad_w, grad_b, strict=True)
        ):
            if has_velocity:
                prev_vw = state.vel_w[idx]
                prev_vb = state.vel_b[idx]
                vw = momentum * prev_vw + step * gw
                vb = momentum * prev_vb + step * gb
            else:
                vw = step * gw
                vb = step * gb

            next_w = w - vw
            next_b = b - vb

            if self.config.inner_update_mode == "stopgrad":
                next_w = next_w.detach()
                next_b = next_b.detach()
                vw = vw.detach()
                vb = vb.detach()

            new_weights.append(next_w)
            new_biases.append(next_b)
            new_vel_w.append(vw)
            new_vel_b.append(vb)

        return DLAState(
            weights=tuple(new_weights),
            biases=tuple(new_biases),
            vel_w=tuple(new_vel_w),
            vel_b=tuple(new_vel_b),
        )

    def mix_states(self, states: Sequence[DLAState], weights: Tensor) -> DLAState:
        if len(states) == 0:
            raise ValueError("states must be non-empty")

        num_layers = len(states[0].weights)
        mixed_weights: list[Tensor] = []
        mixed_biases: list[Tensor] = []
        mixed_vel_w: list[Tensor] = []
        mixed_vel_b: list[Tensor] = []

        for layer_idx in range(num_layers):
            stacked_w = torch.stack([s.weights[layer_idx] for s in states], dim=2)
            stacked_b = torch.stack([s.biases[layer_idx] for s in states], dim=2)

            mixed_w = torch.einsum("bhs,bhsoi->bhoi", weights, stacked_w)
            mixed_b = torch.einsum("bhs,bhso->bho", weights, stacked_b)
            mixed_weights.append(mixed_w)
            mixed_biases.append(mixed_b)

            if states[0].vel_w is not None and states[0].vel_b is not None:
                stacked_vw = torch.stack([s.vel_w[layer_idx] for s in states], dim=2)
                stacked_vb = torch.stack([s.vel_b[layer_idx] for s in states], dim=2)
                mixed_vw = torch.einsum("bhs,bhsoi->bhoi", weights, stacked_vw)
                mixed_vb = torch.einsum("bhs,bhso->bho", weights, stacked_vb)
                mixed_vel_w.append(mixed_vw)
                mixed_vel_b.append(mixed_vb)

        return DLAState(
            weights=tuple(mixed_weights),
            biases=tuple(mixed_biases),
            vel_w=tuple(mixed_vel_w) if mixed_vel_w else None,
            vel_b=tuple(mixed_vel_b) if mixed_vel_b else None,
        )
