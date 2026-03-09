from __future__ import annotations

import pytest
import torch

from memory_caching.loglinear import (
    ChunkedLogLinearAttentionReference,
    LogLinearAttentionReference,
    chunked_loglinear_attention,
    dense_loglinear_attention,
    hierarchical_level_index,
    recurrent_loglinear_attention,
)
from memory_caching.loglinear.chunk_plan import (
    build_chunk_spans,
    classify_pair,
    decompose_dense_loglinear_attention,
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


def test_recurrent_rejects_insufficient_lambda_levels() -> None:
    q = torch.randn(1, 5, 1, 2)
    k = torch.randn(1, 5, 1, 2)
    v = torch.randn(1, 5, 1, 2)
    lambda_levels = torch.randn(1, 5, 1, 2)

    with pytest.raises(ValueError):
        recurrent_loglinear_attention(q, k, v, lambda_levels)


@pytest.mark.parametrize("bad_chunk_size", [0, -1])
def test_chunked_rejects_non_positive_chunk_size(bad_chunk_size: int) -> None:
    q = torch.randn(1, 4, 1, 2)
    k = torch.randn(1, 4, 1, 2)
    v = torch.randn(1, 4, 1, 2)
    lambda_levels = torch.randn(1, 4, 1, 3)

    with pytest.raises(ValueError):
        chunked_loglinear_attention(q, k, v, lambda_levels, chunk_size=bad_chunk_size)


def test_exact_bucket_summary_after_two_steps() -> None:
    q = torch.ones(1, 2, 1, 1)
    k = torch.ones(1, 2, 1, 1)
    v = torch.tensor([[[[2.0]], [[3.0]]]])
    lambda_levels = torch.ones(1, 2, 1, 2)

    _, state = recurrent_loglinear_attention(q, k, v, lambda_levels)
    batch_state = state.batch_states[0]

    assert batch_state.position == 2
    assert len(batch_state.buckets) == 1
    assert batch_state.buckets[0].size == 2
    assert batch_state.buckets[0].start == 0
    assert batch_state.buckets[0].end == 2
    assert torch.allclose(batch_state.buckets[0].summary, torch.tensor([[[5.0]]]))


def test_mask_divergence_is_per_batch_row() -> None:
    q = torch.ones(2, 3, 1, 1)
    k = torch.ones(2, 3, 1, 1)
    v = torch.tensor(
        [
            [[[1.0]], [[2.0]], [[4.0]]],
            [[[1.0]], [[2.0]], [[4.0]]],
        ]
    )
    lambda_levels = torch.ones(2, 3, 1, 3)
    attention_mask = torch.tensor([[True, False, True], [True, True, True]])

    _, state = recurrent_loglinear_attention(q, k, v, lambda_levels, attention_mask=attention_mask)

    assert state.batch_states[0].position == 2
    assert state.batch_states[1].position == 3


def test_chunk_size_sweep_matches_recurrent() -> None:
    torch.manual_seed(5)
    q = torch.randn(1, 8, 2, 3)
    k = torch.randn(1, 8, 2, 3)
    v = torch.randn(1, 8, 2, 4)
    lambda_levels = torch.rand(1, 8, 2, 4)

    recurrent, _ = recurrent_loglinear_attention(q, k, v, lambda_levels)
    for chunk_size in (1, 2, 4, 8):
        chunked, _ = chunked_loglinear_attention(q, k, v, lambda_levels, chunk_size=chunk_size)
        assert torch.allclose(recurrent, chunked, atol=1e-6, rtol=1e-6)


def test_chunked_state_carry_matches_single_pass() -> None:
    torch.manual_seed(6)
    q = torch.randn(1, 7, 1, 2)
    k = torch.randn(1, 7, 1, 2)
    v = torch.randn(1, 7, 1, 3)
    lambda_levels = torch.rand(1, 7, 1, 3)

    full, _ = chunked_loglinear_attention(q, k, v, lambda_levels, chunk_size=3)
    part1, state = chunked_loglinear_attention(q[:, :4], k[:, :4], v[:, :4], lambda_levels[:, :4], chunk_size=3)
    part2, _ = chunked_loglinear_attention(
        q[:, 4:],
        k[:, 4:],
        v[:, 4:],
        lambda_levels[:, 4:],
        chunk_size=3,
        state=state,
    )
    stitched = torch.cat([part1, part2], dim=1)
    assert torch.allclose(full, stitched, atol=1e-6, rtol=1e-6)


def test_zero_length_and_single_token_chunked_cases() -> None:
    q0 = torch.randn(1, 0, 1, 2)
    k0 = torch.randn(1, 0, 1, 2)
    v0 = torch.randn(1, 0, 1, 2)
    lambda0 = torch.randn(1, 0, 1, 1)
    out0, state0 = chunked_loglinear_attention(q0, k0, v0, lambda0, chunk_size=2)
    assert out0.shape == v0.shape
    assert state0 is None

    q1 = torch.ones(1, 1, 1, 1)
    k1 = torch.ones(1, 1, 1, 1)
    v1 = torch.tensor([[[[7.0]]]])
    lambda1 = torch.ones(1, 1, 1, 1)
    out1, state1 = chunked_loglinear_attention(q1, k1, v1, lambda1, chunk_size=2)
    assert out1[0, 0, 0, 0].item() == 7.0
    assert state1.batch_states[0].position == 1


def test_chunked_matches_dense_oracle_on_small_sequence() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 5, 1, 2)
    k = torch.randn(1, 5, 1, 2)
    v = torch.randn(1, 5, 1, 3)
    lambda_levels = torch.rand(1, 5, 1, 3)

    dense = dense_loglinear_attention(q, k, v, lambda_levels)
    chunked, _ = chunked_loglinear_attention(q, k, v, lambda_levels, chunk_size=2)
    assert torch.allclose(dense, chunked, atol=1e-6, rtol=1e-6)


def test_chunk_plan_build_and_classify() -> None:
    spans = build_chunk_spans(seq_len=10, chunk_size=4)
    assert [(span.start, span.end) for span in spans] == [(0, 4), (4, 8), (8, 10)]
    assert classify_pair(1, 3, 4) == "local"
    assert classify_pair(1, 5, 4) == "inter"


def test_dense_decomposition_recombines_to_dense_oracle() -> None:
    torch.manual_seed(8)
    q = torch.randn(1, 6, 2, 3)
    k = torch.randn(1, 6, 2, 3)
    v = torch.randn(1, 6, 2, 4)
    lambda_levels = torch.rand(1, 6, 2, 4)

    dense = dense_loglinear_attention(q, k, v, lambda_levels)
    local, inter = decompose_dense_loglinear_attention(q, k, v, lambda_levels, chunk_size=2)
    assert torch.allclose(dense, local + inter, atol=1e-6, rtol=1e-6)


def test_chunk_boundary_masking_matches_recurrent() -> None:
    torch.manual_seed(9)
    q = torch.randn(1, 6, 1, 3)
    k = torch.randn(1, 6, 1, 3)
    v = torch.randn(1, 6, 1, 2)
    lambda_levels = torch.rand(1, 6, 1, 3)
    attention_mask = torch.tensor([[True, True, False, True, False, True]])

    recurrent, _ = recurrent_loglinear_attention(q, k, v, lambda_levels, attention_mask=attention_mask)
    chunked, _ = chunked_loglinear_attention(q, k, v, lambda_levels, chunk_size=2, attention_mask=attention_mask)
    assert torch.allclose(recurrent, chunked, atol=1e-6, rtol=1e-6)
