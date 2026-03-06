from __future__ import annotations

import torch

from memory_caching import LinearMemoryBackend, MCConfig, MemoryCachingLayer


def main() -> None:
    config = MCConfig(
        d_model=16,
        num_heads=4,
        backend="linear",
        aggregation="grm",
        segment_size=4,
    )
    layer = MemoryCachingLayer(config=config, backend=LinearMemoryBackend())

    x = torch.randn(2, 12, 16)
    y = layer(x)
    print({"input_shape": tuple(x.shape), "output_shape": tuple(y.shape)})


if __name__ == "__main__":
    main()
