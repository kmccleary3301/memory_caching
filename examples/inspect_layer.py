from __future__ import annotations

import torch

from memory_caching import LinearMemoryBackend, MCConfig, MemoryCachingLayer


def main() -> None:
    config = MCConfig(
        d_model=8,
        num_heads=2,
        backend="linear",
        aggregation="grm",
        segment_size=2,
    )
    layer = MemoryCachingLayer(config=config, backend=LinearMemoryBackend())

    x = torch.randn(1, 6, 8)
    y, cache = layer.forward_with_cache(x)
    _, rows = layer.inspect(x)

    print(
        {
            "output_shape": tuple(y.shape),
            "cache_segments": len(cache),
            "first_row_keys": sorted(rows[0].keys()) if rows else [],
        }
    )


if __name__ == "__main__":
    main()
