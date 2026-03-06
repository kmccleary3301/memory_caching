from __future__ import annotations

import torch

from memory_caching.backends.dla import DLABackend, DLAState
from memory_caching.config import DLAConfig, MCConfig
from memory_caching.layer import MemoryCachingLayer


def _make_backend(objective: str = "dot", mode: str = "stopgrad") -> DLABackend:
    return DLABackend(
        DLAConfig(
            memory_width=6,
            memory_depth=2,
            objective=objective,
            inner_update_mode=mode,
            step_size=0.1,
            momentum=0.0,
        )
    )


def _identity(layer: MemoryCachingLayer) -> None:
    eye = torch.eye(layer.config.d_model)
    with torch.no_grad():
        layer.q_proj.weight.copy_(eye)
        layer.k_proj.weight.copy_(eye)
        layer.v_proj.weight.copy_(eye)
        layer.u_proj.weight.copy_(eye)
        layer.o_proj.weight.copy_(eye)


def _randomize_state(state: DLAState, seed: int) -> DLAState:
    torch.manual_seed(seed)
    weights = tuple(torch.randn_like(w) * 0.1 for w in state.weights)
    biases = tuple(torch.randn_like(b) * 0.1 for b in state.biases)
    vel_w = tuple(torch.zeros_like(w) for w in weights)
    vel_b = tuple(torch.zeros_like(b) for b in biases)
    return DLAState(weights=weights, biases=biases, vel_w=vel_w, vel_b=vel_b)


def test_dla_init_state_shapes() -> None:
    backend = _make_backend()
    state = backend.init_state(
        batch_size=2,
        num_heads=3,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert isinstance(state, DLAState)
    assert len(state.weights) == 2
    assert state.weights[0].shape == (2, 3, 6, 4)
    assert state.weights[1].shape == (2, 3, 4, 6)


def test_dla_apply_and_update_sanity() -> None:
    torch.manual_seed(0)
    backend = _make_backend(objective="l2")
    state = backend.init_state(
        batch_size=1,
        num_heads=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    q = torch.randn(1, 2, 4)
    y0 = backend.apply(state, q)

    k = torch.randn(1, 2, 4)
    v = torch.randn(1, 2, 4)
    state2 = backend.update(state, k, v)
    y1 = backend.apply(state2, q)

    assert y0.shape == y1.shape == (1, 2, 4)
    assert not torch.allclose(y0, y1, atol=1e-7)


def test_dla_objective_variants_diverge() -> None:
    torch.manual_seed(1)
    base_state = _make_backend().init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state_template = _randomize_state(base_state, seed=17)

    k = torch.randn(1, 1, 4)
    v = torch.randn(1, 1, 4)

    state_dot = _make_backend(objective="dot").update(state_template, k, v)
    state_l2 = _make_backend(objective="l2").update(state_template, k, v)

    assert not torch.allclose(state_dot.weights[0], state_l2.weights[0], atol=1e-7)


def test_dla_checkpoint_vs_restart_cache_lifecycle() -> None:
    torch.manual_seed(2)
    backend = _make_backend()

    ckp = MemoryCachingLayer(
        config=MCConfig(
            d_model=8,
            num_heads=2,
            backend="dla",
            aggregation="grm",
            segment_size=2,
            state_init_mode="checkpoint",
            dla=backend.config,
        ),
        backend=backend,
    )
    rst = MemoryCachingLayer(
        config=MCConfig(
            d_model=8,
            num_heads=2,
            backend="dla",
            aggregation="grm",
            segment_size=2,
            state_init_mode="restart",
            dla=backend.config,
        ),
        backend=_make_backend(),
    )
    _identity(ckp)
    _identity(rst)

    x = torch.randn(1, 8, 8)
    y_ckp, cache_ckp = ckp.forward_with_cache(x)
    y_rst, cache_rst = rst.forward_with_cache(x)

    assert len(cache_ckp) == len(cache_rst) == 4
    assert not torch.allclose(y_ckp, y_rst, atol=1e-7)


def test_dla_soup_not_forced_equivalent_to_grm_backend_level() -> None:
    torch.manual_seed(3)
    backend = DLABackend(
        DLAConfig(memory_width=8, memory_depth=2, objective="l2", step_size=0.1)
    )

    s0 = _randomize_state(
        backend.init_state(
            batch_size=1,
            num_heads=1,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        seed=123,
    )
    s1 = _randomize_state(
        backend.init_state(
            batch_size=1,
            num_heads=1,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        seed=124,
    )

    q = torch.randn(1, 1, 4)
    weights = torch.tensor([[[0.4, 0.6]]])

    grm_out = 0.4 * backend.apply(s0, q) + 0.6 * backend.apply(s1, q)
    soup_state = backend.mix_states([s0, s1], weights)
    soup_out = backend.apply(soup_state, q)

    assert (grm_out - soup_out).abs().max().item() > 1e-6


def test_dla_update_is_invariant_to_batch_head_replication() -> None:
    torch.manual_seed(31)
    backend = _make_backend(objective="l2", mode="stopgrad")
    base = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    base = _randomize_state(base, seed=77)

    k = torch.randn(1, 1, 4)
    v = torch.randn(1, 1, 4)
    updated_single = backend.update(base, k, v)

    replicated = DLAState(
        weights=tuple(w.repeat(2, 3, 1, 1) for w in base.weights),
        biases=tuple(b.repeat(2, 3, 1) for b in base.biases),
        vel_w=tuple(w.repeat(2, 3, 1, 1) for w in base.vel_w or ()),
        vel_b=tuple(b.repeat(2, 3, 1) for b in base.vel_b or ()),
    )
    k_rep = k.repeat(2, 3, 1)
    v_rep = v.repeat(2, 3, 1)
    updated_rep = backend.update(replicated, k_rep, v_rep)

    expected = updated_single.weights[0][0, 0].view(1, 1, *updated_single.weights[0].shape[-2:])
    assert torch.allclose(updated_rep.weights[0], expected.expand_as(updated_rep.weights[0]), atol=1e-6)


def test_dla_differentiable_mode_unrolls_across_steps() -> None:
    torch.manual_seed(32)
    backend = _make_backend(objective="l2", mode="differentiable")
    state = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state = _randomize_state(state, seed=321)

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


def test_dla_stopgrad_mode_blocks_temporal_gradient_flow() -> None:
    torch.manual_seed(33)
    backend = _make_backend(objective="l2", mode="stopgrad")
    state = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state = _randomize_state(state, seed=322)

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
