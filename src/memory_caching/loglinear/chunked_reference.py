from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .recurrent_reference import (
    LogLinearAttentionReference,
    LogLinearAttentionReferenceConfig,
    recurrent_loglinear_attention,
)
from .state import LogLinearState


def chunked_loglinear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lambda_levels: Tensor,
    *,
    chunk_size: int,
    attention_mask: Tensor | None = None,
    state: LogLinearState | None = None,
) -> tuple[Tensor, LogLinearState]:
    """Correctness-first chunked executor for the recurrent reference path.

    This function processes the sequence in contiguous chunks while carrying the
    recurrent Log-Linear state across chunk boundaries. It is not a claim of
    chunk-scan or fused-kernel parity with the original paper's optimized
    training path.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, received {chunk_size}")

    seq_len = q.shape[1]
    outputs: list[Tensor] = []
    current_state = state

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_mask = None if attention_mask is None else attention_mask[:, start:end]
        chunk_out, current_state = recurrent_loglinear_attention(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            lambda_levels[:, start:end],
            attention_mask=chunk_mask,
            state=current_state,
        )
        outputs.append(chunk_out)

    if outputs:
        return torch.cat(outputs, dim=1), current_state  # type: ignore[return-value]
    return v.new_zeros(v.shape[0], 0, v.shape[2], v.shape[3]), current_state  # type: ignore[return-value]


@dataclass(frozen=True)
class ChunkedLogLinearAttentionReferenceConfig(LogLinearAttentionReferenceConfig):
    chunk_size: int = 128


class ChunkedLogLinearAttentionReference(nn.Module):
    """Chunked correctness-first wrapper for original Log-Linear Attention."""

    def __init__(self, config: ChunkedLogLinearAttentionReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.reference = LogLinearAttentionReference(
            LogLinearAttentionReferenceConfig(
                dim=config.dim,
                heads=config.heads,
                max_levels=config.max_levels,
            )
        )

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
        state: LogLinearState | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, LogLinearState]:
        batch, seq_len, _ = x.shape
        q = self.reference._split_heads(self.reference.q_proj(x))
        k = self.reference._split_heads(self.reference.k_proj(x))
        v = self.reference._split_heads(self.reference.v_proj(x))
        lambda_levels = torch.nn.functional.softplus(
            self.reference.lambda_proj(x).view(
                batch,
                seq_len,
                self.reference.config.heads,
                self.reference.config.max_levels,
            )
        )
        y, next_state = chunked_loglinear_attention(
            q,
            k,
            v,
            lambda_levels,
            chunk_size=self.config.chunk_size,
            attention_mask=attention_mask,
            state=state,
        )
        y = self.reference.out_proj(y.reshape(batch, seq_len, self.reference.config.dim))
        if return_state:
            return y, next_state
        return y
