from __future__ import annotations

import random


def make_seed(base_seed: int, *parts: object) -> int:
    seed = int(base_seed)
    for p in parts:
        seed = (seed * 1315423911 + hash(str(p))) & 0xFFFFFFFF
    random.seed(seed)
    return seed
