from __future__ import annotations


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
