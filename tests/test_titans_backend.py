from __future__ import annotations

import torch

from memory_caching.backends.titans import TitansBackend, TitansState
from memory_caching.config import MCConfig, TitansConfig
from memory_caching.layer import MemoryCachingLayer


def _backend(objective: str = "l2", mode: str = "stopgrad") -> TitansBackend:
    return TitansBackend(
        TitansConfig(
            memory_width=6,
            memory_depth=2,
            objective=objective,
            inner_update_mode=mode,
            step_size=0.1,
            momentum=0.8,
            retention_alpha=0.95,
        )
    )


def _id(layer: MemoryCachingLayer) -> None:
    eye = torch.eye(layer.config.d_model)
    with torch.no_grad():
        layer.q_proj.weight.copy_(eye)
        layer.k_proj.weight.copy_(eye)
        layer.v_proj.weight.copy_(eye)
        layer.u_proj.weight.copy_(eye)
        layer.o_proj.weight.copy_(eye)


def _random_state(state: TitansState, seed: int) -> TitansState:
    torch.manual_seed(seed)
    w = tuple(torch.randn_like(t) * 0.1 for t in state.weights)
    b = tuple(torch.randn_like(t) * 0.1 for t in state.biases)
    sw = tuple(torch.randn_like(t) * 0.01 for t in state.weights)
    sb = tuple(torch.randn_like(t) * 0.01 for t in state.biases)
    return TitansState(weights=w, biases=b, s_w=sw, s_b=sb)


def test_titans_init_shape() -> None:
    backend = _backend()
    st = backend.init_state(
        batch_size=2,
        num_heads=3,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert len(st.weights) == 2
    assert st.weights[0].shape == (2, 3, 6, 4)
    assert st.weights[1].shape == (2, 3, 4, 6)


def test_titans_apply_finite() -> None:
    torch.manual_seed(0)
    backend = _backend()
    st = _random_state(
        backend.init_state(
            batch_size=1,
            num_heads=1,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        seed=10,
    )
    out = backend.apply(st, torch.randn(1, 1, 4))
    assert out.shape == (1, 1, 4)
    assert torch.isfinite(out).all()


def test_titans_update_changes_state() -> None:
    torch.manual_seed(1)
    backend = _backend()
    st = _random_state(
        backend.init_state(
            batch_size=1,
            num_heads=1,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        seed=11,
    )
    st2 = backend.update(st, torch.randn(1, 1, 4), torch.randn(1, 1, 4))
    assert not torch.allclose(st.weights[0], st2.weights[0], atol=1e-7)


def test_titans_objective_variants_diverge() -> None:
    torch.manual_seed(2)
    st = _random_state(
        _backend().init_state(
            batch_size=1,
            num_heads=1,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        seed=12,
    )
    k = torch.randn(1, 1, 4)
    v = torch.randn(1, 1, 4)

    l2 = _backend(objective="l2").update(st, k, v)
    dot = _backend(objective="dot").update(st, k, v)
    assert not torch.allclose(l2.weights[0], dot.weights[0], atol=1e-7)


def test_titans_checkpoint_vs_restart_diverge() -> None:
    torch.manual_seed(3)
    conf_ckp = MCConfig(
        d_model=8,
        num_heads=2,
        backend="titans",
        aggregation="grm",
        segment_size=2,
        state_init_mode="checkpoint",
    )
    conf_rst = MCConfig(
        d_model=8,
        num_heads=2,
        backend="titans",
        aggregation="grm",
        segment_size=2,
        state_init_mode="restart",
    )
    ckp = MemoryCachingLayer(config=conf_ckp, backend=TitansBackend(conf_ckp.titans))
    rst = MemoryCachingLayer(config=conf_rst, backend=TitansBackend(conf_rst.titans))
    _id(ckp)
    _id(rst)
    rst.load_state_dict(ckp.state_dict())

    x = torch.randn(2, 8, 8)
    assert not torch.allclose(ckp(x), rst(x), atol=1e-7)


def test_titans_soup_vs_grm_not_equivalent() -> None:
    torch.manual_seed(4)
    backend = TitansBackend(
        TitansConfig(memory_width=8, memory_depth=2, objective="l2", retention_alpha=0.95)
    )

    s0 = _random_state(
        backend.init_state(
            batch_size=1,
            num_heads=1,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        seed=201,
    )
    s1 = _random_state(
        backend.init_state(
            batch_size=1,
            num_heads=1,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        seed=202,
    )

    q = torch.randn(1, 1, 4)
    w = torch.tensor([[[0.4, 0.6]]])
    grm_out = 0.4 * backend.apply(s0, q) + 0.6 * backend.apply(s1, q)
    soup_state = backend.mix_states([s0, s1], w)
    soup_out = backend.apply(soup_state, q)
    assert (grm_out - soup_out).abs().max().item() > 1e-6


def test_titans_ssc_shape() -> None:
    torch.manual_seed(5)
    conf = MCConfig(
        d_model=8,
        num_heads=2,
        backend="titans",
        aggregation="ssc",
        segment_size=2,
        ssc_top_k=1,
    )
    layer = MemoryCachingLayer(config=conf, backend=TitansBackend(conf.titans))
    _id(layer)
    y, cache = layer(torch.randn(1, 6, 8), return_cache=True)
    assert y.shape == (1, 6, 8)
    assert len(cache) == 3
