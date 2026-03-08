from __future__ import annotations

import torch
from torch import Tensor

from .fenwick import max_active_levels, timestep_buckets


def _validate_shapes(q: Tensor, k: Tensor, v: Tensor, lambda_levels: Tensor) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, T, H, D]")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, received {q.shape} and {k.shape}")
    if q.shape[:3] != v.shape[:3]:
        raise ValueError("q/k and v must agree on [B, T, H]")
    if lambda_levels.ndim != 4:
        raise ValueError("lambda_levels must have shape [B, T, H, L]")

    batch, seq_len, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if lambda_levels.shape[:3] != (batch, seq_len, num_heads):
        raise ValueError("lambda_levels must agree with q/k/v on [B, T, H]")
    if lambda_levels.shape[-1] < max_active_levels(seq_len):
        raise ValueError(
            f"lambda_levels last dim must be >= {max_active_levels(seq_len)} for seq_len={seq_len}, "
            f"received {lambda_levels.shape[-1]}"
        )
    return batch, seq_len, num_heads, key_dim, value_dim


def dense_loglinear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lambda_levels: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Dense oracle for the original Log-Linear Attention reference contract."""

    batch, seq_len, num_heads, _, value_dim = _validate_shapes(q, k, v, lambda_levels)
    out = torch.zeros((batch, seq_len, num_heads, value_dim), dtype=v.dtype, device=v.device)

    if attention_mask is None:
        attention_mask = torch.ones((batch, seq_len), dtype=torch.bool, device=v.device)
    else:
        if attention_mask.shape != (batch, seq_len):
            raise ValueError(
                f"attention_mask must have shape {(batch, seq_len)}, received {tuple(attention_mask.shape)}"
            )
        attention_mask = attention_mask.to(dtype=torch.bool, device=v.device)

    for b in range(batch):
        for t in range(seq_len):
            if not attention_mask[b, t]:
                continue
            q_t = q[b, t]
            for bucket in timestep_buckets(t):
                active = attention_mask[b, bucket.start : bucket.end]
                if not torch.any(active):
                    continue
                k_slice = k[b, bucket.start : bucket.end][active]
                v_slice = v[b, bucket.start : bucket.end][active]
                summary = torch.einsum("shv,shk->hvk", v_slice, k_slice)
                read = torch.einsum("hvk,hk->hv", summary, q_t)
                weight = lambda_levels[b, t, :, bucket.level].unsqueeze(-1)
                out[b, t] = out[b, t] + weight * read

    return out
