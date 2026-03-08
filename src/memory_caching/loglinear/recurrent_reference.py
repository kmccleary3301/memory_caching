from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .dense_oracle import _validate_shapes
from .state import FenwickBucketState, LogLinearState, SingleBatchLogLinearState


def _merge_front_buckets(buckets: list[FenwickBucketState]) -> list[FenwickBucketState]:
    merged = list(buckets)
    while len(merged) >= 2 and merged[0].size == merged[1].size:
        recent = merged[0]
        distant = merged[1]
        merged_bucket = FenwickBucketState(
            summary=recent.summary + distant.summary,
            size=recent.size + distant.size,
            start=distant.start,
            end=recent.end,
        )
        merged = [merged_bucket] + merged[2:]
    return merged


def _empty_state(batch: int) -> LogLinearState:
    return LogLinearState(
        batch_states=tuple(
            SingleBatchLogLinearState(buckets=tuple(), position=0) for _ in range(batch)
        )
    )


def recurrent_loglinear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lambda_levels: Tensor,
    *,
    attention_mask: Tensor | None = None,
    state: LogLinearState | None = None,
) -> tuple[Tensor, LogLinearState]:
    """Step-wise reference implementation for the original Log-Linear contract."""

    batch, seq_len, num_heads, key_dim, value_dim = _validate_shapes(q, k, v, lambda_levels)
    if attention_mask is None:
        attention_mask = torch.ones((batch, seq_len), dtype=torch.bool, device=v.device)
    else:
        if attention_mask.shape != (batch, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(batch, seq_len)}, received {tuple(attention_mask.shape)}"
            )
        attention_mask = attention_mask.to(dtype=torch.bool, device=v.device)

    current = state if state is not None else _empty_state(batch)
    if len(current.batch_states) != batch:
        raise ValueError(
            f"state batch size mismatch: expected {batch}, received {len(current.batch_states)}"
        )

    out = torch.zeros((batch, seq_len, num_heads, value_dim), dtype=v.dtype, device=v.device)
    next_batch_states: list[SingleBatchLogLinearState] = []

    for b in range(batch):
        batch_state = current.batch_states[b]
        buckets = list(batch_state.buckets)
        position = batch_state.position

        for t in range(seq_len):
            if not attention_mask[b, t]:
                continue

            q_t = q[b, t]
            current_bucket = FenwickBucketState(
                summary=torch.einsum("hv,hk->hvk", v[b, t], k[b, t]),
                size=1,
                start=position,
                end=position + 1,
            )
            active_buckets = [current_bucket] + buckets
            for level, bucket in enumerate(active_buckets):
                read = torch.einsum("hvk,hk->hv", bucket.summary, q_t)
                weight = lambda_levels[b, t, :, level].unsqueeze(-1)
                out[b, t] = out[b, t] + weight * read

            buckets = _merge_front_buckets([current_bucket] + buckets)
            position += 1

        next_batch_states.append(
            SingleBatchLogLinearState(buckets=tuple(buckets), position=position)
        )

    return out, LogLinearState(batch_states=tuple(next_batch_states))


@dataclass(frozen=True)
class LogLinearAttentionReferenceConfig:
    dim: int
    heads: int
    max_levels: int


class LogLinearAttentionReference(nn.Module):
    """Correctness-first reference module for original Log-Linear Attention."""

    def __init__(self, config: LogLinearAttentionReferenceConfig) -> None:
        super().__init__()
        if config.dim % config.heads != 0:
            raise ValueError(f"dim={config.dim} must be divisible by heads={config.heads}")
        self.config = config
        self.dim_head = config.dim // config.heads
        self.q_proj = nn.Linear(config.dim, config.dim)
        self.k_proj = nn.Linear(config.dim, config.dim)
        self.v_proj = nn.Linear(config.dim, config.dim)
        self.lambda_proj = nn.Linear(config.dim, config.heads * config.max_levels)
        self.out_proj = nn.Linear(config.dim, config.dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.config.heads, self.dim_head)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
        state: LogLinearState | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, LogLinearState]:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        lambda_levels = torch.nn.functional.softplus(
            self.lambda_proj(x).view(batch, seq_len, self.config.heads, self.config.max_levels)
        )
        y, next_state = recurrent_loglinear_attention(
            q,
            k,
            v,
            lambda_levels,
            attention_mask=attention_mask,
            state=state,
        )
        y = self.out_proj(y.reshape(batch, seq_len, self.config.dim))
        if return_state:
            return y, next_state
        return y
