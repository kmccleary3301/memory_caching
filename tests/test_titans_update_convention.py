from __future__ import annotations

import torch

from memory_caching.backends.titans import TitansBackend, TitansState
from memory_caching.config import TitansConfig


def _make_backend(convention: str) -> TitansBackend:
    return TitansBackend(
        TitansConfig(
            memory_width=1,
            memory_depth=2,
            objective="l2",
            inner_update_mode="stopgrad",
            step_size=0.1,
            momentum=0.3,
            retention_alpha=0.8,
            update_convention=convention,  # type: ignore[arg-type]
        )
    )


def _state_with_known_second_layer_buffer(backend: TitansBackend) -> TitansState:
    state = backend.init_state(
        batch_size=1,
        num_heads=1,
        head_dim=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Keep all parameters at zero so pred=0 in this setup.
    w = tuple(torch.zeros_like(t) for t in state.weights)
    b = tuple(torch.zeros_like(t) for t in state.biases)
    sw = tuple(torch.zeros_like(t) for t in state.s_w or ())
    sb = list(torch.zeros_like(t) for t in state.s_b or ())
    # Set a known previous optimizer buffer only on the final bias.
    sb[-1] = torch.full_like(sb[-1], 0.25)
    return TitansState(weights=w, biases=b, s_w=sw, s_b=tuple(sb))


def _run_one_step(backend: TitansBackend) -> TitansState:
    state = _state_with_known_second_layer_buffer(backend)
    # With k=0 and zero params, pred=0. For l2 and target v=1, grad on final
    # bias is hand-derived as: d/d b2 (pred-target)^2 = -2.
    k = torch.zeros(1, 1, 1)
    v = torch.ones(1, 1, 1)
    return backend.update(state, k, v)


def test_titans_paper_convention_matches_hand_derived_bias_update() -> None:
    backend = _make_backend("paper")
    updated = _run_one_step(backend)

    beta = backend.config.momentum
    eta = backend.config.step_size
    alpha = backend.config.retention_alpha
    prev_s = 0.25
    grad = -2.0

    expected_s = beta * prev_s - eta * grad
    expected_b = alpha * 0.0 - expected_s

    actual_s = float(updated.s_b[-1].item())
    actual_b = float(updated.biases[-1].item())

    assert abs(actual_s - expected_s) < 1e-6
    assert abs(actual_b - expected_b) < 1e-6


def test_titans_gradient_descent_convention_matches_hand_derived_bias_update() -> None:
    backend = _make_backend("gradient_descent")
    updated = _run_one_step(backend)

    beta = backend.config.momentum
    eta = backend.config.step_size
    alpha = backend.config.retention_alpha
    prev_s = 0.25
    grad = -2.0

    expected_s = beta * prev_s + eta * grad
    expected_b = alpha * 0.0 - expected_s

    actual_s = float(updated.s_b[-1].item())
    actual_b = float(updated.biases[-1].item())

    assert abs(actual_s - expected_s) < 1e-6
    assert abs(actual_b - expected_b) < 1e-6
