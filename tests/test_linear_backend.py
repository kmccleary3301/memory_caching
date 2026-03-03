from __future__ import annotations

import torch

from memory_caching.backends.linear import LinearMemoryBackend


def test_linear_backend_init_state_shape() -> None:
    backend = LinearMemoryBackend()
    state = backend.init_state(
        batch_size=2,
        num_heads=3,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert state.shape == (2, 3, 4, 4)


def test_linear_backend_update_and_apply_match_manual_math() -> None:
    backend = LinearMemoryBackend()

    state = torch.zeros(1, 1, 2, 2)
    k_t = torch.tensor([[[2.0, 3.0]]])
    v_t = torch.tensor([[[5.0, 7.0]]])
    q_t = torch.tensor([[[11.0, 13.0]]])

    updated = backend.update(state, k_t, v_t)
    expected_state = torch.tensor([[[[10.0, 15.0], [14.0, 21.0]]]])

    assert torch.allclose(updated, expected_state, atol=1e-6)

    output = backend.apply(updated, q_t)
    expected_output = torch.einsum("bhde,bhe->bhd", expected_state, q_t)
    assert torch.allclose(output, expected_output, atol=1e-6)


def test_linear_backend_mix_states_weighted_sum() -> None:
    backend = LinearMemoryBackend()

    s1 = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    s2 = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]]]])
    weights = torch.tensor([[[0.25, 0.75]]])

    mixed = backend.mix_states([s1, s2], weights)
    expected = torch.tensor([[[[1.75, 0.0], [0.0, 1.75]]]])

    assert torch.allclose(mixed, expected, atol=1e-6)
