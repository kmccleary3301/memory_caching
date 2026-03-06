from __future__ import annotations

import hashlib


def make_seed(base_seed: int, *parts: object) -> int:
    seed = int(base_seed) & 0xFFFFFFFF
    for part in parts:
        payload = repr(part).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=4).digest()
        part_hash = int.from_bytes(digest, byteorder="little", signed=False)
        seed = (seed * 1315423911 + part_hash) & 0xFFFFFFFF
    return seed
