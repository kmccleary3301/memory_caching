from __future__ import annotations

import torch

from memory_caching.models import build_tiny_model_from_spec


def test_build_tiny_loglinear_ref_model() -> None:
    model = build_tiny_model_from_spec(
        {
            "model_family": "tiny_loglinear_ref_lm",
            "vocab_size": 32,
            "d_model": 16,
            "num_heads": 4,
            "loglinear_max_levels": 4,
        }
    )
    tokens = torch.randint(0, 32, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 32)


def test_build_tiny_loglinear_chunked_model() -> None:
    model = build_tiny_model_from_spec(
        {
            "model_family": "tiny_loglinear_chunked_lm",
            "vocab_size": 32,
            "d_model": 16,
            "num_heads": 4,
            "loglinear_max_levels": 4,
            "loglinear_chunk_size": 4,
        }
    )
    tokens = torch.randint(0, 32, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 32)
