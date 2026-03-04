from __future__ import annotations

import pytest

from memory_caching.smoke import run_smoke_eval


@pytest.mark.parametrize("backend", ["linear", "dla", "titans"])
def test_smoke_eval_deterministic_per_backend(backend: str) -> None:
    base = dict(
        warmup_steps=1,
        batch_size=1,
        seq_len=8,
        vocab_size=16,
        d_model=8,
        num_heads=2,
        device="cpu",
        seed=123,
        backend=backend,
    )
    a = run_smoke_eval(**base)
    b = run_smoke_eval(**base)
    assert a == b


def test_smoke_eval_changes_with_seed() -> None:
    common = dict(
        warmup_steps=1,
        batch_size=1,
        seq_len=8,
        vocab_size=16,
        d_model=8,
        num_heads=2,
        device="cpu",
        backend="linear",
    )
    a = run_smoke_eval(**common, seed=1)
    b = run_smoke_eval(**common, seed=2)
    assert float(a["eval_loss"]) != float(b["eval_loss"])
