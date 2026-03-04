from __future__ import annotations

import pytest
import torch

from memory_caching.config import MCConfig
from memory_caching.layer import MemoryCachingLayer


class _BadShapeBackend:
    def init_state(self, *, batch_size, num_heads, head_dim, device, dtype):
        return {"ok": True}

    def update(self, state, k_t, v_t):
        return state

    def apply(self, state, q_t):
        return q_t[..., :-1]


class _BadDtypeBackend:
    def init_state(self, *, batch_size, num_heads, head_dim, device, dtype):
        return {"ok": True}

    def update(self, state, k_t, v_t):
        return state

    def apply(self, state, q_t):
        return q_t.to(torch.float64)


class _BadDeviceBackend:
    def init_state(self, *, batch_size, num_heads, head_dim, device, dtype):
        return {"ok": True}

    def update(self, state, k_t, v_t):
        return state

    def apply(self, state, q_t):
        return torch.empty_like(q_t, device=torch.device("meta"))


class _NoneUpdateStateBackend:
    def init_state(self, *, batch_size, num_heads, head_dim, device, dtype):
        return {"ok": True}

    def update(self, state, k_t, v_t):
        return None

    def apply(self, state, q_t):
        return q_t


class _NoneInitStateBackend:
    def init_state(self, *, batch_size, num_heads, head_dim, device, dtype):
        return None

    def update(self, state, k_t, v_t):
        return state

    def apply(self, state, q_t):
        return q_t


def _cfg() -> MCConfig:
    return MCConfig(
        d_model=8,
        num_heads=2,
        backend="linear",
        aggregation="grm",
        segment_size=2,
    )


def test_backend_guard_invalid_shape_raises() -> None:
    layer = MemoryCachingLayer(config=_cfg(), backend=_BadShapeBackend())
    with pytest.raises(RuntimeError, match="aggregate_output expected shape"):
        _ = layer(torch.randn(1, 4, 8))


def test_backend_guard_invalid_dtype_raises() -> None:
    layer = MemoryCachingLayer(config=_cfg(), backend=_BadDtypeBackend())
    with pytest.raises(RuntimeError, match="aggregate_output expected dtype"):
        _ = layer(torch.randn(1, 4, 8))


def test_backend_guard_invalid_device_raises() -> None:
    layer = MemoryCachingLayer(config=_cfg(), backend=_BadDeviceBackend())
    with pytest.raises(RuntimeError, match="expected device|expected device cpu"):
        _ = layer(torch.randn(1, 4, 8))


def test_backend_guard_none_update_state_raises() -> None:
    layer = MemoryCachingLayer(config=_cfg(), backend=_NoneUpdateStateBackend())
    with pytest.raises(RuntimeError, match="backend returned None state at stage=update"):
        _ = layer(torch.randn(1, 4, 8))


def test_backend_guard_none_init_state_raises() -> None:
    layer = MemoryCachingLayer(config=_cfg(), backend=_NoneInitStateBackend())
    with pytest.raises(RuntimeError, match="backend returned None state at stage=init_state"):
        _ = layer(torch.randn(1, 4, 8))
