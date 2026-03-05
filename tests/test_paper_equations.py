from __future__ import annotations

import pytest
import torch

from memory_caching.backends.dla import DLABackend
from memory_caching.backends.linear import LinearMemoryBackend
from memory_caching.config import DLAConfig, MCConfig
from memory_caching.contracts import SegmentCache
from memory_caching.layer import MemoryCachingLayer
from memory_caching.segmentation import logarithmic_segments


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    torch.manual_seed(0)


def _make_linear_layer(aggregation: str, *, ssc_top_k: int = 1) -> MemoryCachingLayer:
    return MemoryCachingLayer(
        config=MCConfig(
            d_model=2,
            num_heads=1,
            backend="linear",
            aggregation=aggregation,  # type: ignore[arg-type]
            segment_size=2,
            ssc_top_k=ssc_top_k,
        ),
        backend=LinearMemoryBackend(),
    )


def _state(diag0: float, diag1: float) -> torch.Tensor:
    out = torch.zeros(1, 1, 2, 2)
    out[0, 0, 0, 0] = diag0
    out[0, 0, 1, 1] = diag1
    return out


def _cache(state: torch.Tensor, ctx0: float, ctx1: float) -> SegmentCache:
    return SegmentCache(
        mem_state=state,
        seg_context=torch.tensor([[[ctx0, ctx1]]], dtype=torch.float32),
        seg_len=2,
    )


def _identity_projections(layer: MemoryCachingLayer) -> None:
    eye = torch.eye(layer.config.d_model)
    with torch.no_grad():
        layer.q_proj.weight.copy_(eye)
        layer.k_proj.weight.copy_(eye)
        layer.v_proj.weight.copy_(eye)
        layer.u_proj.weight.copy_(eye)
        layer.o_proj.weight.copy_(eye)


def test_rm_equation_matches_sum_of_cached_plus_online() -> None:
    layer = _make_linear_layer("residual")
    q_t = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)
    online = _state(1.0, 1.0)
    cached = [
        _cache(_state(4.0, 0.0), 1.0, 0.0),
        _cache(_state(0.0, 5.0), 0.0, 1.0),
    ]

    expected = layer.backend.apply(online, q_t)
    expected = expected + layer.backend.apply(cached[0].mem_state, q_t)
    expected = expected + layer.backend.apply(cached[1].mem_state, q_t)

    actual = layer._residual_aggregate(q_t=q_t, online_state=online, cached=cached)
    assert torch.allclose(actual, expected, atol=1e-6), "RM equation mismatch"


def test_grm_gamma_weights_match_softmax_dot_formula() -> None:
    layer = _make_linear_layer("grm")
    q_t = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    u_t = torch.tensor([[[2.0, -1.0]]], dtype=torch.float32)
    online = _state(1.0, 2.0)
    cached = [
        _cache(_state(2.0, 0.0), 1.0, 0.0),
        _cache(_state(0.0, 3.0), -1.0, 2.0),
    ]
    online_context = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)

    contexts = [cached[0].seg_context, cached[1].seg_context, online_context]
    scores = torch.stack([torch.einsum("bhd,bhd->bh", u_t, c) for c in contexts], dim=-1)
    weights = torch.softmax(scores, dim=-1)

    responses = [
        layer.backend.apply(cached[0].mem_state, q_t),
        layer.backend.apply(cached[1].mem_state, q_t),
        layer.backend.apply(online, q_t),
    ]
    expected = (
        weights[..., 0].unsqueeze(-1) * responses[0]
        + weights[..., 1].unsqueeze(-1) * responses[1]
        + weights[..., 2].unsqueeze(-1) * responses[2]
    )
    actual = layer._grm_or_soup_aggregate(
        q_t=q_t,
        u_t=u_t,
        online_state=online,
        cached=cached,
        online_context=online_context,
    )
    assert torch.allclose(actual, expected, atol=1e-6), "GRM gamma softmax formula mismatch"


def test_soup_equals_grm_for_linear_backend_equation_case() -> None:
    grm = _make_linear_layer("grm")
    soup = _make_linear_layer("soup")
    q_t = torch.tensor([[[1.0, -1.0]]], dtype=torch.float32)
    u_t = torch.tensor([[[0.7, 0.2]]], dtype=torch.float32)
    online = _state(2.0, 1.0)
    cached = [
        _cache(_state(1.0, 3.0), 0.2, 0.1),
        _cache(_state(3.0, 1.0), 0.4, -0.1),
    ]
    online_context = torch.tensor([[[0.1, 0.3]]], dtype=torch.float32)

    y_grm = grm._grm_or_soup_aggregate(
        q_t=q_t,
        u_t=u_t,
        online_state=online,
        cached=cached,
        online_context=online_context,
    )
    y_soup = soup._grm_or_soup_aggregate(
        q_t=q_t,
        u_t=u_t,
        online_state=online,
        cached=cached,
        online_context=online_context,
    )
    assert torch.allclose(y_grm, y_soup, atol=1e-6), "Linear-memory Soup must equal GRM"


def test_ssc_top_k_masking_matches_analytic_selection() -> None:
    ssc = _make_linear_layer("ssc", ssc_top_k=1)
    q_t = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
    u_t = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    online = _state(1.0, 1.0)
    cached = [
        _cache(_state(5.0, 0.0), 0.2, 0.0),  # score 0.2
        _cache(_state(0.0, 5.0), 2.0, 0.0),  # score 2.0 <- selected
        _cache(_state(3.0, 3.0), -1.0, 0.0),  # score -1.0
    ]
    online_context = torch.tensor([[[0.5, 0.0]]], dtype=torch.float32)

    selected_score = torch.tensor([[[2.0]]], dtype=torch.float32)
    online_score = torch.tensor([[[0.5]]], dtype=torch.float32)
    weights = torch.softmax(torch.cat([selected_score, online_score], dim=-1), dim=-1)
    expected = (
        weights[..., 0].unsqueeze(-1) * ssc.backend.apply(cached[1].mem_state, q_t)
        + weights[..., 1].unsqueeze(-1) * ssc.backend.apply(online, q_t)
    )
    actual = ssc._ssc_aggregate(
        q_t=q_t,
        u_t=u_t,
        online_state=online,
        cached=cached,
        online_context=online_context,
    )
    assert torch.allclose(actual, expected, atol=1e-6), "SSC top-k routing mismatch"


def test_checkpoint_vs_restart_diverge_for_nonlinear_backend() -> None:
    torch.manual_seed(2026)
    dla_cfg = DLAConfig(memory_width=4, memory_depth=2, objective="l2", step_size=0.05)
    checkpoint = MemoryCachingLayer(
        config=MCConfig(
            d_model=8,
            num_heads=2,
            backend="dla",
            aggregation="grm",
            segment_size=2,
            state_init_mode="checkpoint",
            dla=dla_cfg,
        ),
        backend=DLABackend(dla_cfg),
    )
    restart = MemoryCachingLayer(
        config=MCConfig(
            d_model=8,
            num_heads=2,
            backend="dla",
            aggregation="grm",
            segment_size=2,
            state_init_mode="restart",
            dla=dla_cfg,
        ),
        backend=DLABackend(dla_cfg),
    )
    _identity_projections(checkpoint)
    _identity_projections(restart)
    restart.load_state_dict(checkpoint.state_dict())

    x = torch.randn(1, 8, 8)
    y_checkpoint = checkpoint(x)
    y_restart = restart(x)
    assert not torch.allclose(y_checkpoint, y_restart, atol=1e-6), "Checkpoint/restart should diverge"


def test_logarithmic_segmentation_equation_example_37() -> None:
    assert logarithmic_segments(37) == [32, 4, 1], "Log decomposition must match paper example 37 -> [32,4,1]"
