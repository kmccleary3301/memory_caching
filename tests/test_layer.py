from __future__ import annotations

import torch

from memory_caching.backends.linear import LinearMemoryBackend
from memory_caching.config import MCConfig
from memory_caching.layer import MemoryCachingLayer


def _make_layer(
    *,
    aggregation: str,
    segment_size: int,
    segmentation: str = "constant",
) -> MemoryCachingLayer:
    config = MCConfig(
        d_model=8,
        num_heads=2,
        aggregation=aggregation,
        segmentation=segmentation,
        segment_size=segment_size,
        ssc_top_k=1,
    )
    layer = MemoryCachingLayer(config=config, backend=LinearMemoryBackend())
    _set_identity_projections(layer)
    return layer


def _set_identity_projections(layer: MemoryCachingLayer) -> None:
    eye = torch.eye(layer.config.d_model)
    with torch.no_grad():
        layer.q_proj.weight.copy_(eye)
        layer.k_proj.weight.copy_(eye)
        layer.v_proj.weight.copy_(eye)
        layer.u_proj.weight.copy_(eye)
        layer.o_proj.weight.copy_(eye)


def test_layer_causality_no_future_leak() -> None:
    torch.manual_seed(0)
    layer = _make_layer(aggregation="grm", segment_size=3)

    x = torch.randn(2, 9, 8)
    y_ref = layer(x)

    x_perturbed = x.clone()
    x_perturbed[:, 7:, :] = x_perturbed[:, 7:, :] + 10.0
    y_perturbed = layer(x_perturbed)

    assert torch.allclose(y_ref[:, :7, :], y_perturbed[:, :7, :], atol=1e-6)


def test_layer_segmentation_determinism_with_explicit_lengths() -> None:
    torch.manual_seed(1)
    layer = _make_layer(aggregation="residual", segment_size=3)

    x = torch.randn(1, 9, 8)
    y_default, cache_default = layer(x, return_cache=True)
    y_explicit, cache_explicit = layer(x, segment_lengths=[3, 3, 3], return_cache=True)

    assert torch.allclose(y_default, y_explicit, atol=1e-6)
    assert [c.seg_len for c in cache_default] == [3, 3, 3]
    assert [c.seg_len for c in cache_explicit] == [3, 3, 3]


def test_linear_memory_grm_and_soup_outputs_match() -> None:
    torch.manual_seed(2)
    grm = _make_layer(aggregation="grm", segment_size=2)
    soup = _make_layer(aggregation="soup", segment_size=2)
    soup.load_state_dict(grm.state_dict())

    x = torch.randn(2, 8, 8)
    y_grm = grm(x)
    y_soup = soup(x)

    assert torch.allclose(y_grm, y_soup, atol=1e-6)


def test_single_segment_residual_equals_grm() -> None:
    torch.manual_seed(3)
    residual = _make_layer(aggregation="residual", segment_size=32)
    grm = _make_layer(aggregation="grm", segment_size=32)
    grm.load_state_dict(residual.state_dict())

    x = torch.randn(2, 6, 8)
    y_residual = residual(x)
    y_grm = grm(x)

    assert torch.allclose(y_residual, y_grm, atol=1e-6)


def test_ssc_forward_shape_and_cache_count() -> None:
    torch.manual_seed(4)
    ssc = _make_layer(aggregation="ssc", segment_size=2)

    x = torch.randn(1, 6, 8)
    y, cache = ssc(x, return_cache=True)

    assert y.shape == x.shape
    assert len(cache) == 3
