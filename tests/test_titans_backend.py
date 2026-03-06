from __future__ import annotations

import torch

from memory_caching.backends.titans import TitansBackend, TitansState
from memory_caching.config import MCConfig, TitansConfig
from memory_caching.layer import MemoryCachingLayer


def _backend(
    objective: str = "l2",
    mode: str = "stopgrad",
    convention: str = "paper",
) -> TitansBackend:
    return TitansBackend(
        TitansConfig(
            memory_width=6,
            memory_depth=2,
            objective=objective,
            inner_update_mode=mode,
            step_size=0.1,
            momentum=0.8,
            retention_alpha=0.95,
            update_convention=convention,
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
    y, cache = layer.forward_with_cache(torch.randn(1, 6, 8))
    assert y.shape == (1, 6, 8)
    assert len(cache) == 3


def test_titans_update_is_invariant_to_batch_head_replication() -> None:
    torch.manual_seed(41)
    backend = _backend(objective="l2", mode="stopgrad", convention="paper")
    base = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    base = _random_state(base, seed=301)

    k = torch.randn(1, 1, 4)
    v = torch.randn(1, 1, 4)
    updated_single = backend.update(base, k, v)

    replicated = TitansState(
        weights=tuple(w.repeat(2, 3, 1, 1) for w in base.weights),
        biases=tuple(b.repeat(2, 3, 1) for b in base.biases),
        s_w=tuple(sw.repeat(2, 3, 1, 1) for sw in base.s_w or ()),
        s_b=tuple(sb.repeat(2, 3, 1) for sb in base.s_b or ()),
    )
    k_rep = k.repeat(2, 3, 1)
    v_rep = v.repeat(2, 3, 1)
    updated_rep = backend.update(replicated, k_rep, v_rep)

    expected = updated_single.weights[0][0, 0].view(1, 1, *updated_single.weights[0].shape[-2:])
    assert torch.allclose(updated_rep.weights[0], expected.expand_as(updated_rep.weights[0]), atol=1e-6)


def test_titans_paper_update_convention_matches_equation() -> None:
    torch.manual_seed(42)
    backend = _backend(objective="l2", mode="stopgrad", convention="paper")
    state = _random_state(
        backend.init_state(
            batch_size=1,
            num_heads=1,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        seed=302,
    )
    k = torch.randn(1, 1, 4)
    v = torch.randn(1, 1, 4)

    with torch.enable_grad():
        w_vars = [w.detach().requires_grad_(True) for w in state.weights]
        b_vars = [b.detach().requires_grad_(True) for b in state.biases]
        tmp_state = TitansState(weights=tuple(w_vars), biases=tuple(b_vars))
        pred = backend.apply(tmp_state, k)
        loss = backend._loss(pred, v)
        grads = torch.autograd.grad(loss, [*w_vars, *b_vars], allow_unused=False)

    n = len(w_vars)
    grad_w = grads[:n]
    grad_b = grads[n:]
    updated = backend.update(state, k, v)

    beta = backend.config.momentum
    eta = backend.config.step_size
    alpha = backend.config.retention_alpha

    for idx in range(n):
        expected_sw = beta * state.s_w[idx] - eta * grad_w[idx]
        expected_sb = beta * state.s_b[idx] - eta * grad_b[idx]
        expected_w = alpha * w_vars[idx] - expected_sw
        expected_b = alpha * b_vars[idx] - expected_sb
        assert torch.allclose(updated.weights[idx], expected_w.detach(), atol=1e-6)
        assert torch.allclose(updated.biases[idx], expected_b.detach(), atol=1e-6)


def test_titans_differentiable_mode_unrolls_across_steps() -> None:
    torch.manual_seed(43)
    backend = _backend(objective="l2", mode="differentiable", convention="paper")
    state = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state = _random_state(state, seed=401)

    k1 = torch.randn(1, 1, 4, requires_grad=True)
    v1 = torch.randn(1, 1, 4)
    k2 = torch.randn(1, 1, 4)
    v2 = torch.randn(1, 1, 4)
    q = torch.randn(1, 1, 4)

    s1 = backend.update(state, k1, v1)
    s2 = backend.update(s1, k2, v2)
    loss = backend.apply(s2, q).sum()
    grad_k1 = torch.autograd.grad(loss, k1, allow_unused=True)[0]

    assert grad_k1 is not None
    assert grad_k1.abs().sum().item() > 0.0


def test_titans_stopgrad_mode_blocks_temporal_gradient_flow() -> None:
    torch.manual_seed(44)
    backend = _backend(objective="l2", mode="stopgrad", convention="paper")
    state = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state = _random_state(state, seed=402)

    k1 = torch.randn(1, 1, 4, requires_grad=True)
    v1 = torch.randn(1, 1, 4)
    k2 = torch.randn(1, 1, 4)
    v2 = torch.randn(1, 1, 4)
    q = torch.randn(1, 1, 4)

    s1 = backend.update(state, k1, v1)
    s2 = backend.update(s1, k2, v2)
    loss = backend.apply(s2, q).sum()
    if not loss.requires_grad:
        return
    grad_k1 = torch.autograd.grad(loss, k1, allow_unused=True)[0]

    assert grad_k1 is None or grad_k1.abs().sum().item() == 0.0
