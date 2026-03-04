from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .config import MCConfig, StateInitMode
from .contracts import (
    MemoryBackend,
    MixableMemoryBackend,
    SegmentCache,
    ensure_backend_state,
    ensure_head_tensor,
)
from .segmentation import (
    constant_segments,
    logarithmic_segments,
    spans_from_lengths,
    validate_lengths,
)
from .tree import tree_clone, tree_detach_clone

Tensor = torch.Tensor


class MemoryCachingLayer(nn.Module):
    """
    Core Memory Caching layer that wraps a recurrent memory backend.

    Input/Output shape: [B, T, D].
    """

    def __init__(self, *, config: MCConfig, backend: MemoryBackend) -> None:
        super().__init__()
        if not isinstance(backend, MemoryBackend):
            raise TypeError("backend must satisfy MemoryBackend protocol")

        self.config = config
        self.backend = backend
        self._backend_supports_state_mixing = isinstance(backend, MixableMemoryBackend)
        self.head_dim = config.d_model // config.num_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.u_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        return x.reshape(bsz, seq_len, self.config.num_heads, self.head_dim)

    def _merge_heads(self, x: Tensor) -> Tensor:
        bsz, seq_len, num_heads, head_dim = x.shape
        return x.reshape(bsz, seq_len, num_heads * head_dim)

    def _assert_projected_tensor(
        self,
        *,
        name: str,
        value: Tensor,
        batch_size: int,
        seq_len: int,
    ) -> None:
        expected = (batch_size, seq_len, self.config.num_heads, self.head_dim)
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"{name} expected shape {expected}, got {tuple(value.shape)}"
            )

    def _build_segment_spans(
        self,
        *,
        seq_len: int,
        segment_size: int | None,
        segment_lengths: Sequence[int] | None,
    ) -> list[tuple[int, int]]:
        if segment_lengths is not None:
            lengths = validate_lengths(segment_lengths, total_length=seq_len)
            return spans_from_lengths(lengths)

        if self.config.segmentation == "constant":
            size = self.config.segment_size if segment_size is None else segment_size
            if size <= 0:
                raise ValueError("segment_size must be positive")
            return constant_segments(seq_len, size)

        if self.config.segmentation == "logarithmic":
            lengths = logarithmic_segments(seq_len)
            return spans_from_lengths(lengths)

        raise ValueError(f"Unsupported segmentation mode: {self.config.segmentation}")

    def _snapshot_state(self, state: object) -> object:
        if self.config.detach_cached_states:
            return tree_detach_clone(state)
        return tree_clone(state)

    def _snapshot_context(self, context: Tensor) -> Tensor:
        if self.config.detach_cached_context:
            return context.detach().clone()
        return context.clone()

    def _context_scores(self, u_t: Tensor, contexts: list[Tensor]) -> Tensor:
        ctx = torch.stack(contexts, dim=2)  # [B,H,S,Dh]
        scores = torch.einsum("bhd,bhsd->bhs", u_t, ctx)
        if self.config.softmax_temperature != 1.0:
            scores = scores / self.config.softmax_temperature
        return scores

    def _residual_aggregate(
        self,
        *,
        q_t: Tensor,
        online_state: object,
        cached: list[SegmentCache],
    ) -> Tensor:
        out = self.backend.apply(online_state, q_t)
        for cache_entry in cached:
            out = out + self.backend.apply(cache_entry.mem_state, q_t)
        return out

    def _grm_or_soup_aggregate(
        self,
        *,
        q_t: Tensor,
        u_t: Tensor,
        online_state: object,
        cached: list[SegmentCache],
        online_context: Tensor,
    ) -> Tensor:
        contexts = [c.seg_context for c in cached] + [online_context]
        scores = self._context_scores(u_t, contexts)
        weights = torch.softmax(scores, dim=-1)

        if self.config.aggregation == "soup" and self._backend_supports_state_mixing:
            states = [c.mem_state for c in cached] + [online_state]
            mixed_state = self.backend.mix_states(states, weights)
            return self.backend.apply(mixed_state, q_t)

        responses = [self.backend.apply(c.mem_state, q_t) for c in cached]
        responses.append(self.backend.apply(online_state, q_t))
        stacked_responses = torch.stack(responses, dim=2)  # [B,H,S,Dh]
        return torch.einsum("bhs,bhsd->bhd", weights, stacked_responses)

    def _ssc_aggregate(
        self,
        *,
        q_t: Tensor,
        u_t: Tensor,
        online_state: object,
        cached: list[SegmentCache],
        online_context: Tensor,
    ) -> Tensor:
        if len(cached) == 0:
            return self.backend.apply(online_state, q_t)

        cached_contexts = [c.seg_context for c in cached]
        cached_scores = self._context_scores(u_t, cached_contexts)

        k = min(self.config.ssc_top_k, len(cached))
        topk_indices = torch.topk(cached_scores, k=k, dim=-1).indices
        mask = torch.zeros_like(cached_scores, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)

        masked_cached_scores = cached_scores.masked_fill(~mask, float("-inf"))
        online_scores = torch.einsum("bhd,bhd->bh", u_t, online_context)
        if self.config.softmax_temperature != 1.0:
            online_scores = online_scores / self.config.softmax_temperature

        all_scores = torch.cat(
            [masked_cached_scores, online_scores.unsqueeze(-1)], dim=-1
        )
        all_weights = torch.softmax(all_scores, dim=-1)

        cached_weights = all_weights[..., :-1]
        online_weight = all_weights[..., -1].unsqueeze(-1)

        out = online_weight * self.backend.apply(online_state, q_t)
        for idx, cache_entry in enumerate(cached):
            if torch.any(mask[..., idx]):
                out = out + cached_weights[..., idx].unsqueeze(-1) * self.backend.apply(
                    cache_entry.mem_state, q_t
                )
        return out

    def _aggregate_token(
        self,
        *,
        q_t: Tensor,
        u_t: Tensor,
        online_state: object,
        cached: list[SegmentCache],
        online_context: Tensor,
    ) -> Tensor:
        if self.config.aggregation == "residual":
            return self._residual_aggregate(
                q_t=q_t,
                online_state=online_state,
                cached=cached,
            )

        if self.config.aggregation in {"grm", "soup"}:
            return self._grm_or_soup_aggregate(
                q_t=q_t,
                u_t=u_t,
                online_state=online_state,
                cached=cached,
                online_context=online_context,
            )

        if self.config.aggregation == "ssc":
            return self._ssc_aggregate(
                q_t=q_t,
                u_t=u_t,
                online_state=online_state,
                cached=cached,
                online_context=online_context,
            )

        raise ValueError(f"Unsupported aggregation variant: {self.config.aggregation}")

    def forward(
        self,
        x: Tensor,
        *,
        segment_size: int | None = None,
        segment_lengths: Sequence[int] | None = None,
        state_init_mode: StateInitMode | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, list[SegmentCache]]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, D]")
        if x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"expected d_model={self.config.d_model}, got {x.shape[-1]}"
            )

        bsz, seq_len, _ = x.shape
        effective_init_mode = (
            self.config.state_init_mode if state_init_mode is None else state_init_mode
        )

        spans = self._build_segment_spans(
            seq_len=seq_len,
            segment_size=segment_size,
            segment_lengths=segment_lengths,
        )

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        u = self._split_heads(self.u_proj(x))
        self._assert_projected_tensor(name="q", value=q, batch_size=bsz, seq_len=seq_len)
        self._assert_projected_tensor(name="k", value=k, batch_size=bsz, seq_len=seq_len)
        self._assert_projected_tensor(name="v", value=v, batch_size=bsz, seq_len=seq_len)
        self._assert_projected_tensor(name="u", value=u, batch_size=bsz, seq_len=seq_len)

        outputs = torch.empty_like(q)
        cached: list[SegmentCache] = []
        prev_final_state: object | None = None

        for start, end in spans:
            if effective_init_mode == "checkpoint" and prev_final_state is not None:
                online_state = prev_final_state
            elif effective_init_mode in {"checkpoint", "restart"}:
                online_state = self.backend.init_state(
                    batch_size=bsz,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                raise ValueError(f"Unsupported state_init_mode: {effective_init_mode}")
            online_state = ensure_backend_state(online_state, stage="init_state")

            online_context = torch.zeros(
                bsz,
                self.config.num_heads,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )
            ensure_head_tensor(
                "online_context",
                online_context,
                batch_size=bsz,
                num_heads=self.config.num_heads,
                head_dim=self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )

            for t in range(start, end):
                k_t = k[:, t]
                v_t = v[:, t]
                q_t = q[:, t]
                u_t = q_t if self.config.use_q_as_u else u[:, t]
                ensure_head_tensor(
                    "k_t",
                    k_t,
                    batch_size=bsz,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
                ensure_head_tensor(
                    "v_t",
                    v_t,
                    batch_size=bsz,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
                ensure_head_tensor(
                    "q_t",
                    q_t,
                    batch_size=bsz,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
                ensure_head_tensor(
                    "u_t",
                    u_t,
                    batch_size=bsz,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )

                online_state = self.backend.update(online_state, k_t, v_t)
                online_state = ensure_backend_state(online_state, stage="update")
                online_context = online_context + k_t
                ensure_head_tensor(
                    "online_context",
                    online_context,
                    batch_size=bsz,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )

                out_t = self._aggregate_token(
                    q_t=q_t,
                    u_t=u_t,
                    online_state=online_state,
                    cached=cached,
                    online_context=online_context,
                )
                ensure_head_tensor(
                    "aggregate_output",
                    out_t,
                    batch_size=bsz,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
                outputs[:, t] = out_t

            ensure_head_tensor(
                "segment_context",
                online_context,
                batch_size=bsz,
                num_heads=self.config.num_heads,
                head_dim=self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )
            cached.append(
                SegmentCache(
                    mem_state=self._snapshot_state(online_state),
                    seg_context=self._snapshot_context(online_context),
                    seg_len=end - start,
                )
            )
            prev_final_state = online_state

        y = self.o_proj(self._merge_heads(outputs))
        if return_cache:
            return y, cached
        return y
