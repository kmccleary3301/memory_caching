from __future__ import annotations

import torch

from memory_caching.loglinear import ChunkedLogLinearAttentionReference
from memory_caching.loglinear.chunked_reference import ChunkedLogLinearAttentionReferenceConfig


def main() -> None:
    x = torch.randn(2, 12, 16)
    layer = ChunkedLogLinearAttentionReference(
        ChunkedLogLinearAttentionReferenceConfig(dim=16, heads=4, max_levels=4, chunk_size=4)
    )
    y = layer(x)
    print({"input_shape": tuple(x.shape), "output_shape": tuple(y.shape), "chunk_size": 4})


if __name__ == "__main__":
    main()
