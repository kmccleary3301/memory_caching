from __future__ import annotations

import pytest

from memory_caching.segmentation import (
    constant_segments,
    logarithmic_segments,
    spans_from_lengths,
    validate_lengths,
)


def test_constant_segments_even_division() -> None:
    assert constant_segments(12, 4) == [(0, 4), (4, 8), (8, 12)]


def test_constant_segments_with_tail() -> None:
    assert constant_segments(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_logarithmic_segments_binary_decomposition() -> None:
    assert logarithmic_segments(37) == [32, 4, 1]


def test_spans_from_lengths() -> None:
    assert spans_from_lengths([3, 2, 5]) == [(0, 3), (3, 5), (5, 10)]


def test_validate_lengths_total_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        validate_lengths([1, 2], total_length=4)
