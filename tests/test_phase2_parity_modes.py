from __future__ import annotations

import pytest
import torch

from memory_caching.backends.dla import DLABackend
from memory_caching.backends.linear import LinearMemoryBackend
from memory_caching.backends.titans import TitansBackend
from memory_caching.config import DLAConfig, MCConfig, TitansConfig
from memory_caching.layer import MemoryCachingLayer


class _LinearNoMixBackend:
    def __init__(self) -> None:
        self._impl = LinearMemoryBackend()

    def init_state(self, *, batch_size, num_heads, head_dim, device, dtype):
        return self._impl.init_state(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

    def update(self, state, k_t, v_t):
        return self._impl.update(state, k_t, v_t)

    def apply(self, state, q_t):
        return self._impl.apply(state, q_t)


def _id(layer: MemoryCachingLayer) -> None:
    eye = torch.eye(layer.config.d_model)
    with torch.no_grad():
        layer.q_proj.weight.copy_(eye)
        layer.k_proj.weight.copy_(eye)
        layer.v_proj.weight.copy_(eye)
        layer.u_proj.weight.copy_(eye)
        layer.o_proj.weight.copy_(eye)


@pytest.mark.parametrize("aggregation", ["residual", "grm", "soup", "ssc"])
def test_checkpoint_vs_restart_diverge_for_all_aggregations(aggregation: str) -> None:
    torch.manual_seed(7)
    cfg_a = MCConfig(
        d_model=8,
        num_heads=2,
        backend="linear",
        aggregation=aggregation,
        segment_size=2,
        ssc_top_k=1,
        state_init_mode="checkpoint",
    )
    cfg_b = MCConfig(
        d_model=8,
        num_heads=2,
        backend="linear",
        aggregation=aggregation,
        segment_size=2,
        ssc_top_k=1,
        state_init_mode="restart",
    )
    a = MemoryCachingLayer(config=cfg_a, backend=LinearMemoryBackend())
    b = MemoryCachingLayer(config=cfg_b, backend=LinearMemoryBackend())
    _id(a)
    _id(b)
    b.load_state_dict(a.state_dict())

    x = torch.randn(1, 8, 8)
    assert not torch.allclose(a(x), b(x), atol=1e-7)


def test_soup_fallback_matches_grm_for_nonmixable_backend() -> None:
    torch.manual_seed(11)
    cfg_soup = MCConfig(
        d_model=8,
        num_heads=2,
        backend="linear",
        aggregation="soup",
        segment_size=2,
        state_init_mode="checkpoint",
        allow_output_mixture_fallback=True,
    )
    cfg_grm = MCConfig(
        d_model=8,
        num_heads=2,
        backend="linear",
        aggregation="grm",
        segment_size=2,
        state_init_mode="checkpoint",
    )
    soup = MemoryCachingLayer(config=cfg_soup, backend=_LinearNoMixBackend())
    grm = MemoryCachingLayer(config=cfg_grm, backend=_LinearNoMixBackend())
    _id(soup)
    _id(grm)
    grm.load_state_dict(soup.state_dict())

    x = torch.randn(1, 8, 8)
    assert torch.allclose(soup(x), grm(x), atol=1e-7)


def test_dla_update_mode_graph_properties_diverge() -> None:
    torch.manual_seed(13)
    stop = DLABackend(
        DLAConfig(
            memory_width=8,
            memory_depth=2,
            objective="dot",
            inner_update_mode="stopgrad",
        )
    )
    diff = DLABackend(
        DLAConfig(
            memory_width=8,
            memory_depth=2,
            objective="dot",
            inner_update_mode="differentiable",
        )
    )
    st = stop.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    k = torch.randn(1, 1, 4)
    v = torch.randn(1, 1, 4)
    s_stop = stop.update(st, k, v)
    s_diff = diff.update(st, k, v)

    assert not s_stop.weights[0].requires_grad
    assert s_diff.weights[0].requires_grad


def test_titans_update_mode_graph_properties_diverge() -> None:
    torch.manual_seed(17)
    stop = TitansBackend(
        TitansConfig(
            memory_width=8,
            memory_depth=2,
            objective="l2",
            inner_update_mode="stopgrad",
        )
    )
    diff = TitansBackend(
        TitansConfig(
            memory_width=8,
            memory_depth=2,
            objective="l2",
            inner_update_mode="differentiable",
        )
    )
    st = stop.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    k = torch.randn(1, 1, 4)
    v = torch.randn(1, 1, 4)
    s_stop = stop.update(st, k, v)
    s_diff = diff.update(st, k, v)

    assert not s_stop.weights[0].requires_grad
    assert s_diff.weights[0].requires_grad
