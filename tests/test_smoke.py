from __future__ import annotations

import json

import pytest

from memory_caching.smoke import run_smoke_eval, run_smoke_train


def test_smoke_train_validation_invalid_backend() -> None:
    with pytest.raises(ValueError):
        run_smoke_train(steps=1, seq_len=8, backend="bad")


def test_smoke_train_validation_invalid_aggregation() -> None:
    with pytest.raises(ValueError):
        run_smoke_train(steps=1, seq_len=8, aggregation="bad")


def test_smoke_train_validation_invalid_segmentation() -> None:
    with pytest.raises(ValueError):
        run_smoke_train(steps=1, seq_len=8, segmentation="bad")


def test_smoke_train_validation_invalid_state_mode() -> None:
    with pytest.raises(ValueError):
        run_smoke_train(steps=1, seq_len=8, state_init_mode="bad")


def test_smoke_eval_writes_json_with_stable_keys(tmp_path) -> None:
    out = tmp_path / "eval_metrics.json"
    metrics = run_smoke_eval(
        warmup_steps=1,
        batch_size=1,
        seq_len=8,
        vocab_size=16,
        d_model=8,
        num_heads=2,
        out_json=str(out),
    )
    on_disk = json.loads(out.read_text())

    expected_keys = {
        "mode",
        "device",
        "backend",
        "steps",
        "batch_size",
        "seq_len",
        "vocab_size",
        "initial_loss",
        "final_loss",
        "eval_loss",
        "eval_accuracy",
        "cache_segments",
        "mean_segment_len",
        "trainable_params",
    }
    assert expected_keys.issubset(on_disk.keys())
    assert on_disk == metrics


def test_smoke_dla_path_runs() -> None:
    metrics = run_smoke_eval(
        warmup_steps=1,
        batch_size=1,
        seq_len=8,
        vocab_size=16,
        d_model=8,
        num_heads=2,
        backend="dla",
        dla_memory_width=8,
        dla_memory_depth=2,
        dla_objective="dot",
        dla_inner_update_mode="stopgrad",
    )
    assert metrics["backend"] == "dla"


def test_smoke_titans_path_runs_and_schema_parity() -> None:
    metrics = run_smoke_eval(
        warmup_steps=1,
        batch_size=1,
        seq_len=8,
        vocab_size=16,
        d_model=8,
        num_heads=2,
        backend="titans",
        titans_memory_width=8,
        titans_memory_depth=2,
        titans_objective="l2",
        titans_inner_update_mode="stopgrad",
    )
    expected_keys = {
        "mode",
        "device",
        "backend",
        "steps",
        "batch_size",
        "seq_len",
        "vocab_size",
        "initial_loss",
        "final_loss",
        "eval_loss",
        "eval_accuracy",
        "cache_segments",
        "mean_segment_len",
        "trainable_params",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["backend"] == "titans"
