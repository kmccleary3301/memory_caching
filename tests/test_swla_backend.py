from __future__ import annotations

import torch

from memory_caching.backends.swla import SWLABackend, SWLAState
from memory_caching.config import SWLAConfig


def _backend(alpha: float = 0.7, beta: float = 0.2, lam: float = 0.5) -> SWLABackend:
    return SWLABackend(SWLAConfig(alpha=alpha, beta=beta, lam=lam))


def test_swla_init_state_shapes() -> None:
    backend = _backend()
    state = backend.init_state(
        batch_size=2,
        num_heads=3,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert isinstance(state, SWLAState)
    assert state.m_prev.shape == (2, 3, 4, 4)
    assert state.m_prev2.shape == (2, 3, 4, 4)


def test_swla_recurrence_matches_reference_equation() -> None:
    torch.manual_seed(101)
    backend = _backend(alpha=0.75, beta=0.15, lam=0.4)
    state0 = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    k1 = torch.randn(1, 1, 4)
    v1 = torch.randn(1, 1, 4)
    k2 = torch.randn(1, 1, 4)
    v2 = torch.randn(1, 1, 4)

    state1 = backend.update(state0, k1, v1)
    outer1 = torch.einsum("bhd,bhe->bhde", v1, k1)
    expected1 = backend.config.lam * outer1
    assert torch.allclose(state1.m_prev, expected1, atol=1e-6)
    assert torch.allclose(state1.m_prev2, state0.m_prev, atol=1e-6)

    state2 = backend.update(state1, k2, v2)
    outer2 = torch.einsum("bhd,bhe->bhde", v2, k2)
    expected2 = (
        backend.config.alpha * state1.m_prev
        + backend.config.beta * state1.m_prev2
        + backend.config.lam * outer2
    )
    assert torch.allclose(state2.m_prev, expected2, atol=1e-6)
    assert torch.allclose(state2.m_prev2, state1.m_prev, atol=1e-6)


def test_swla_mix_states_matches_response_space_mixture() -> None:
    torch.manual_seed(102)
    backend = _backend(alpha=0.9, beta=0.1, lam=0.8)

    s0 = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    s1 = backend.update(s0, torch.randn(1, 1, 4), torch.randn(1, 1, 4))
    s2 = backend.update(s1, torch.randn(1, 1, 4), torch.randn(1, 1, 4))

    q = torch.randn(1, 1, 4)
    weights = torch.tensor([[[0.35, 0.65]]], dtype=torch.float32)

    grm_out = 0.35 * backend.apply(s1, q) + 0.65 * backend.apply(s2, q)
    soup_state = backend.mix_states([s1, s2], weights)
    soup_out = backend.apply(soup_state, q)
    assert torch.allclose(grm_out, soup_out, atol=1e-6)


def test_swla_update_is_invariant_to_batch_head_replication() -> None:
    torch.manual_seed(103)
    backend = _backend(alpha=0.8, beta=-0.1, lam=0.6)
    base = SWLAState(
        m_prev=torch.randn(1, 1, 4, 4),
        m_prev2=torch.randn(1, 1, 4, 4),
    )
    k = torch.randn(1, 1, 4)
    v = torch.randn(1, 1, 4)

    updated_single = backend.update(base, k, v)

    replicated = SWLAState(
        m_prev=base.m_prev.repeat(2, 3, 1, 1),
        m_prev2=base.m_prev2.repeat(2, 3, 1, 1),
    )
    k_rep = k.repeat(2, 3, 1)
    v_rep = v.repeat(2, 3, 1)
    updated_rep = backend.update(replicated, k_rep, v_rep)

    expected_prev = updated_single.m_prev[0, 0].view(1, 1, 4, 4)
    expected_prev2 = updated_single.m_prev2[0, 0].view(1, 1, 4, 4)
    assert torch.allclose(updated_rep.m_prev, expected_prev.expand_as(updated_rep.m_prev), atol=1e-6)
    assert torch.allclose(updated_rep.m_prev2, expected_prev2.expand_as(updated_rep.m_prev2), atol=1e-6)
