from __future__ import annotations

import torch

from memory_caching.backends.linear import LinearMemoryBackend
from memory_caching.backends.swla import SWLABackend
from memory_caching.config import MCConfig, SWLAConfig
from memory_caching.layer import MemoryCachingLayer


class DummyNoMixBackend:
    def init_state(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, num_heads, head_dim, head_dim, device=device, dtype=dtype)

    def update(self, state: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        return state + torch.einsum("bhd,bhe->bhde", v_t, k_t)

    def apply(self, state: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bhde,bhe->bhd", state, q_t)


def _make_layer(
    *,
    aggregation: str,
    segment_size: int,
    segmentation: str = "constant",
    state_init_mode: str = "checkpoint",
    ssc_top_k: int = 1,
    allow_output_mixture_fallback: bool = False,
    backend: object | None = None,
) -> MemoryCachingLayer:
    config = MCConfig(
        d_model=8,
        num_heads=2,
        aggregation=aggregation,
        segmentation=segmentation,
        segment_size=segment_size,
        state_init_mode=state_init_mode,
        ssc_top_k=ssc_top_k,
        allow_output_mixture_fallback=allow_output_mixture_fallback,
    )
    layer = MemoryCachingLayer(
        config=config,
        backend=backend if backend is not None else LinearMemoryBackend(),
    )
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
    y_default, cache_default = layer.forward_with_cache(x)
    y_explicit, cache_explicit = layer.forward_with_cache(x, segment_lengths=[3, 3, 3])

    assert torch.allclose(y_default, y_explicit, atol=1e-6)
    assert [c.seg_len for c in cache_default] == [3, 3, 3]
    assert [c.seg_len for c in cache_explicit] == [3, 3, 3]


def test_non_uniform_segment_lengths_cache_accounting() -> None:
    torch.manual_seed(11)
    layer = _make_layer(aggregation="grm", segment_size=9)
    x = torch.randn(1, 10, 8)
    _, cache = layer.forward_with_cache(x, segment_lengths=[2, 3, 5])
    assert [c.seg_len for c in cache] == [2, 3, 5]


def test_linear_memory_grm_and_soup_outputs_match() -> None:
    torch.manual_seed(2)
    grm = _make_layer(aggregation="grm", segment_size=2)
    soup = _make_layer(aggregation="soup", segment_size=2)
    soup.load_state_dict(grm.state_dict())

    x = torch.randn(2, 8, 8)
    y_grm = grm(x)
    y_soup = soup(x)

    assert torch.allclose(y_grm, y_soup, atol=1e-6)


def test_swla_memory_grm_and_soup_outputs_match() -> None:
    torch.manual_seed(12)
    swla_cfg = SWLAConfig(alpha=0.9, beta=0.1, lam=1.0)
    grm = MemoryCachingLayer(
        config=MCConfig(
            d_model=8,
            num_heads=2,
            backend="swla",
            aggregation="grm",
            segment_size=2,
            swla=swla_cfg,
        ),
        backend=SWLABackend(swla_cfg),
    )
    soup = MemoryCachingLayer(
        config=MCConfig(
            d_model=8,
            num_heads=2,
            backend="swla",
            aggregation="soup",
            segment_size=2,
            swla=swla_cfg,
        ),
        backend=SWLABackend(swla_cfg),
    )
    _set_identity_projections(grm)
    _set_identity_projections(soup)
    soup.load_state_dict(grm.state_dict())

    x = torch.randn(2, 8, 8)
    y_grm = grm(x)
    y_soup = soup(x)

    assert torch.allclose(y_grm, y_soup, atol=1e-6)


def test_soup_fallback_without_mixable_backend_matches_grm() -> None:
    torch.manual_seed(42)
    backend = DummyNoMixBackend()
    grm = _make_layer(
        aggregation="grm",
        segment_size=2,
        allow_output_mixture_fallback=True,
        backend=backend,
    )
    soup = _make_layer(
        aggregation="soup",
        segment_size=2,
        allow_output_mixture_fallback=True,
        backend=backend,
    )
    soup.load_state_dict(grm.state_dict())

    x = torch.randn(2, 8, 8)
    y_grm = grm(x)
    y_soup = soup(x)
    assert torch.allclose(y_grm, y_soup, atol=1e-6)


def test_soup_without_mixable_backend_raises_without_fallback_flag() -> None:
    torch.manual_seed(43)
    backend = DummyNoMixBackend()
    soup = _make_layer(
        aggregation="soup",
        segment_size=2,
        allow_output_mixture_fallback=False,
        backend=backend,
    )
    x = torch.randn(1, 4, 8)
    try:
        soup(x)
    except ValueError as exc:
        assert "allow_output_mixture_fallback" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-mixable soup without fallback")


def test_single_segment_residual_equals_grm() -> None:
    torch.manual_seed(3)
    residual = _make_layer(aggregation="residual", segment_size=32)
    grm = _make_layer(aggregation="grm", segment_size=32)
    grm.load_state_dict(residual.state_dict())

    x = torch.randn(2, 6, 8)
    y_residual = residual(x)
    y_grm = grm(x)

    assert torch.allclose(y_residual, y_grm, atol=1e-6)


def test_checkpoint_and_restart_modes_diverge_multisegment() -> None:
    torch.manual_seed(14)
    checkpoint = _make_layer(
        aggregation="grm",
        segment_size=2,
        state_init_mode="checkpoint",
    )
    restart = _make_layer(
        aggregation="grm",
        segment_size=2,
        state_init_mode="restart",
    )
    restart.load_state_dict(checkpoint.state_dict())

    x = torch.randn(2, 8, 8)
    y_checkpoint = checkpoint(x)
    y_restart = restart(x)

    assert not torch.allclose(y_checkpoint, y_restart, atol=1e-6)


def test_ssc_forward_shape_and_cache_count() -> None:
    torch.manual_seed(4)
    ssc = _make_layer(aggregation="ssc", segment_size=2, ssc_top_k=1)

    x = torch.randn(1, 6, 8)
    y, cache = ssc.forward_with_cache(x)

    assert y.shape == x.shape
    assert len(cache) == 3


def test_ssc_top_k_ge_cache_matches_grm() -> None:
    torch.manual_seed(7)
    ssc = _make_layer(aggregation="ssc", segment_size=2, ssc_top_k=10)
    grm = _make_layer(aggregation="grm", segment_size=2)
    ssc.load_state_dict(grm.state_dict())

    x = torch.randn(2, 8, 8)
    y_ssc = ssc(x)
    y_grm = grm(x)

    assert torch.allclose(y_ssc, y_grm, atol=1e-6)


def test_ssc_single_segment_matches_grm() -> None:
    torch.manual_seed(8)
    ssc = _make_layer(aggregation="ssc", segment_size=100, ssc_top_k=1)
    grm = _make_layer(aggregation="grm", segment_size=100)
    ssc.load_state_dict(grm.state_dict())

    x = torch.randn(2, 8, 8)
    assert torch.allclose(ssc(x), grm(x), atol=1e-6)


def test_attention_mask_blocks_updates_and_zeros_masked_positions() -> None:
    torch.manual_seed(21)
    layer = _make_layer(aggregation="grm", segment_size=2)
    x = torch.randn(1, 6, 8)

    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    y_masked = layer(x, attention_mask=attention_mask)
    y_prefix = layer(x[:, :3, :])

    assert torch.allclose(y_masked[:, :3, :], y_prefix, atol=1e-6)
    assert torch.allclose(y_masked[:, 3:, :], torch.zeros_like(y_masked[:, 3:, :]), atol=1e-6)
