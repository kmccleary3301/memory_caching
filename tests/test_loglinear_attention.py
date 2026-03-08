from __future__ import annotations

import torch

from memory_caching.loglinear import (
    ChunkedLogLinearAttentionReference,
    LogLinearAttentionReference,
    chunked_loglinear_attention,
    dense_loglinear_attention,
    hierarchical_level_index,
    recurrent_loglinear_attention,
)
from memory_caching.loglinear.chunked_reference import ChunkedLogLinearAttentionReferenceConfig
from memory_caching.loglinear.recurrent_reference import LogLinearAttentionReferenceConfig


def test_hierarchical_level_index_t4() -> None:
    actual = hierarchical_level_index(4)
    expected = torch.tensor(
        [
            [0, -1, -1, -1],
            [1, 0, -1, -1],
            [1, 1, 0, -1],
            [2, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    assert torch.equal(actual, expected)


def test_dense_matches_recurrent_reference() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 6, 3, 4)
    k = torch.randn(2, 6, 3, 4)
    v = torch.randn(2, 6, 3, 5)
    lambda_levels = torch.rand(2, 6, 3, 4)

    dense = dense_loglinear_attention(q, k, v, lambda_levels)
    recurrent, _ = recurrent_loglinear_attention(q, k, v, lambda_levels)

    assert torch.allclose(dense, recurrent, atol=1e-6, rtol=1e-6)


def test_equal_lambdas_collapse_to_linear_attention_sum() -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 5, 2, 3)
    k = torch.randn(1, 5, 2, 3)
    v = torch.randn(1, 5, 2, 4)
    lambda_levels = torch.ones(1, 5, 2, 3)

    actual = dense_loglinear_attention(q, k, v, lambda_levels)

    expected = torch.zeros_like(actual)
    for t in range(q.shape[1]):
        summary = torch.einsum("bshv,bshk->bhvk", v[:, : t + 1], k[:, : t + 1])
        expected[:, t] = torch.einsum("bhvk,bhk->bhv", summary, q[:, t])

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_masked_tokens_do_not_update_state_or_output() -> None:
    torch.manual_seed(2)
    q = torch.randn(1, 4, 1, 3)
    k = torch.randn(1, 4, 1, 3)
    v = torch.randn(1, 4, 1, 2)
    lambda_levels = torch.rand(1, 4, 1, 3)
    attention_mask = torch.tensor([[True, False, True, True]])

    out, state = recurrent_loglinear_attention(q, k, v, lambda_levels, attention_mask=attention_mask)

    assert torch.equal(out[:, 1], torch.zeros_like(out[:, 1]))
    assert state.batch_states[0].position == 3


def test_state_carry_matches_full_forward() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 6, 2, 3)
    k = torch.randn(1, 6, 2, 3)
    v = torch.randn(1, 6, 2, 4)
    lambda_levels = torch.rand(1, 6, 2, 4)

    full, _ = recurrent_loglinear_attention(q, k, v, lambda_levels)
    part1, state = recurrent_loglinear_attention(q[:, :3], k[:, :3], v[:, :3], lambda_levels[:, :3])
    part2, _ = recurrent_loglinear_attention(
        q[:, 3:],
        k[:, 3:],
        v[:, 3:],
        lambda_levels[:, 3:],
        state=state,
    )
    stitched = torch.cat([part1, part2], dim=1)

    assert torch.allclose(full, stitched, atol=1e-6, rtol=1e-6)


def test_reference_module_forward_shape() -> None:
    module = LogLinearAttentionReference(
        LogLinearAttentionReferenceConfig(dim=16, heads=4, max_levels=4)
    )
    x = torch.randn(2, 8, 16)
    y = module(x)
    assert y.shape == x.shape


def test_dense_manual_hand_case_t3() -> None:
    q = torch.ones(1, 4, 1, 1)
    k = torch.ones(1, 4, 1, 1)
    v = torch.tensor([[[[1.0]], [[2.0]], [[4.0]], [[8.0]]]])
    lambda_levels = torch.zeros(1, 4, 1, 4)
    lambda_levels[0, 3, 0, 0] = 10.0
    lambda_levels[0, 3, 0, 1] = 100.0
    lambda_levels[0, 3, 0, 2] = 1000.0

    out = dense_loglinear_attention(q, k, v, lambda_levels)

    assert out[0, 3, 0, 0].item() == 3480.0


def test_chunked_matches_recurrent_reference() -> None:
    torch.manual_seed(4)
    q = torch.randn(2, 9, 3, 4)
    k = torch.randn(2, 9, 3, 4)
    v = torch.randn(2, 9, 3, 5)
    lambda_levels = torch.rand(2, 9, 3, 5)
    attention_mask = torch.tensor(
        [
            [True, True, True, False, True, True, True, True, False],
            [True, False, True, True, True, False, True, True, True],
        ]
    )

    recurrent, recurrent_state = recurrent_loglinear_attention(
        q,
        k,
        v,
        lambda_levels,
        attention_mask=attention_mask,
    )
    chunked, chunked_state = chunked_loglinear_attention(
        q,
        k,
        v,
        lambda_levels,
        chunk_size=3,
        attention_mask=attention_mask,
    )

    assert torch.allclose(recurrent, chunked, atol=1e-6, rtol=1e-6)
    assert len(recurrent_state.batch_states) == len(chunked_state.batch_states)
    for lhs, rhs in zip(recurrent_state.batch_states, chunked_state.batch_states):
        assert lhs.position == rhs.position
        assert len(lhs.buckets) == len(rhs.buckets)
        for lhs_bucket, rhs_bucket in zip(lhs.buckets, rhs.buckets):
            assert lhs_bucket.size == rhs_bucket.size
            assert lhs_bucket.start == rhs_bucket.start
            assert lhs_bucket.end == rhs_bucket.end
            assert torch.allclose(lhs_bucket.summary, rhs_bucket.summary, atol=1e-6, rtol=1e-6)


def test_chunked_reference_module_forward_shape() -> None:
    module = ChunkedLogLinearAttentionReference(
        ChunkedLogLinearAttentionReferenceConfig(dim=16, heads=4, max_levels=4, chunk_size=4)
    )
    x = torch.randn(2, 8, 16)
    y = module(x)
    assert y.shape == x.shape
