from __future__ import annotations

from typing import Sequence


def constant_segments(length: int, segment_size: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")

    segments: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + segment_size, length)
        segments.append((start, end))
        start = end
    return segments


def logarithmic_segments(length: int) -> list[int]:
    if length <= 0:
        return []

    result: list[int] = []
    remaining = length
    power = 1
    while power * 2 <= remaining:
        power *= 2

    while remaining > 0:
        while power > remaining:
            power //= 2
        result.append(power)
        remaining -= power
    return result


def validate_lengths(
    lengths: Sequence[int], *, total_length: int | None = None
) -> list[int]:
    validated: list[int] = []
    for idx, value in enumerate(lengths):
        if value <= 0:
            raise ValueError(f"segment length at index {idx} must be positive")
        validated.append(int(value))

    if total_length is not None and sum(validated) != total_length:
        raise ValueError(
            f"sum of segment lengths must equal {total_length}, got {sum(validated)}"
        )
    return validated


def spans_from_lengths(lengths: Sequence[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for length in validate_lengths(lengths):
        end = start + length
        spans.append((start, end))
        start = end
    return spans
