from __future__ import annotations

from dataclasses import dataclass

import torch

from memory_caching.loglinear.fenwick import hierarchical_level_index


@dataclass(frozen=True)
class ChunkSpan:
    chunk_index: int
    start: int
    end: int


def build_chunk_spans(seq_len: int, chunk_size: int) -> list[ChunkSpan]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    spans: list[ChunkSpan] = []
    for start in range(0, seq_len, chunk_size):
        spans.append(ChunkSpan(chunk_index=len(spans), start=start, end=min(seq_len, start + chunk_size)))
    return spans


def classify_pair(source_index: int, target_index: int, chunk_size: int) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return "local" if source_index // chunk_size == target_index // chunk_size else "inter"


def decompose_dense_loglinear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lambda_levels: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, and v must agree on batch/sequence/head dimensions")
    if lambda_levels.shape[:3] != q.shape[:3]:
        raise ValueError("lambda_levels must agree with q on batch/sequence/head dimensions")

    batch, seq_len, heads, _ = q.shape
    value_dim = v.shape[-1]
    levels = hierarchical_level_index(seq_len).to(device=q.device)
    local = torch.zeros(batch, seq_len, heads, value_dim, device=q.device, dtype=q.dtype)
    inter = torch.zeros_like(local)

    for t in range(seq_len):
        q_t = q[:, t]
        for s in range(t + 1):
            level = int(levels[t, s].item())
            if level < 0:
                continue
            lam = lambda_levels[:, t, :, level]
            score = (q_t * k[:, s]).sum(dim=-1)
            contribution = lam.unsqueeze(-1) * score.unsqueeze(-1) * v[:, s]
            if classify_pair(s, t, chunk_size) == "local":
                local[:, t] = local[:, t] + contribution
            else:
                inter[:, t] = inter[:, t] + contribution
    return local, inter
