from __future__ import annotations

import torch

import memory_caching
from memory_caching import LinearMemoryBackend, MCConfig, MemoryCachingLayer, SegmentCache


def _layer() -> MemoryCachingLayer:
    return MemoryCachingLayer(
        config=MCConfig(d_model=8, num_heads=2, backend="linear", aggregation="grm", segment_size=2),
        backend=LinearMemoryBackend(),
    )


def test_root_surface_exposes_core_modules_only() -> None:
    assert hasattr(memory_caching, "MemoryCachingLayer")
    assert hasattr(memory_caching, "SegmentCache")
    assert not hasattr(memory_caching, "run_smoke_train")
    assert not hasattr(memory_caching, "run_smoke_eval")


def test_forward_returns_tensor_only() -> None:
    layer = _layer()
    x = torch.randn(1, 6, 8)
    y = layer(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape


def test_forward_with_cache_returns_segment_cache_objects() -> None:
    layer = _layer()
    x = torch.randn(1, 6, 8)
    y, cache = layer.forward_with_cache(x)
    assert y.shape == x.shape
    assert all(isinstance(entry, SegmentCache) for entry in cache)


def test_inspect_returns_debug_rows() -> None:
    layer = _layer()
    x = torch.randn(1, 6, 8)
    y, rows = layer.inspect(x)
    assert y.shape == x.shape
    assert isinstance(rows, list)
    assert rows
    assert isinstance(rows[0], dict)
