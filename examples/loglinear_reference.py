from __future__ import annotations

import torch

from memory_caching.loglinear import LogLinearAttentionReference
from memory_caching.loglinear.recurrent_reference import LogLinearAttentionReferenceConfig


def main() -> None:
    x = torch.randn(2, 8, 16)
    layer = LogLinearAttentionReference(
        LogLinearAttentionReferenceConfig(dim=16, heads=4, max_levels=4)
    )
    y = layer(x)
    print({"input_shape": tuple(x.shape), "output_shape": tuple(y.shape)})


if __name__ == "__main__":
    main()
