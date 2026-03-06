from __future__ import annotations

import json

import torch
from typer.testing import CliRunner

from memory_caching.cli import app
from memory_caching.models import build_tiny_model_from_spec


def _write_tiny_mc_checkpoint(path) -> str:
    model_spec = {
        "model_family": "tiny_mc_lm",
        "vocab_size": 256,
        "d_model": 8,
        "num_heads": 2,
        "backend": "linear",
        "aggregation": "grm",
        "segment_size": 2,
    }
    model = build_tiny_model_from_spec(model_spec)
    checkpoint = path / "tiny_mc.pt"
    torch.save(
        {
            "model_spec": model_spec,
            "model_state": model.state_dict(),
            "global_step": 1,
        },
        checkpoint,
    )
    return str(checkpoint)


def test_bench_niah_warns_for_rule_based_adapter_and_emits_adapter_type() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "bench",
                "niah",
                "--adapter",
                "linear",
                "--tasks",
                "s_niah_1",
                "--context-lengths",
                "128",
                "--samples-per-length",
                "1",
                "--seed",
                "0",
                "--out-dir",
                "outputs/benchmarks/test_cli_niah",
            ],
        )

    assert result.exit_code == 0
    assert "WARNING: benchmark adapters are rule-based compatibility adapters" in result.output
    json_start = result.output.find("{")
    assert json_start >= 0
    payload = json.loads(result.output[json_start:])
    assert payload["adapter_type"] == "rule_based"


def test_debug_layer_emits_schema_and_non_empty_rows() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "debug-layer",
                "--backend",
                "linear",
                "--aggregation",
                "grm",
                "--batch-size",
                "1",
                "--seq-len",
                "4",
                "--d-model",
                "8",
                "--num-heads",
                "2",
                "--out-json",
                "outputs/debug/debug_layer.json",
            ],
        )

        assert result.exit_code == 0
        json_start = result.output.find("{")
        assert json_start >= 0
        payload = json.loads(result.output[json_start:])
        assert payload["mode"] == "debug_layer"
        assert payload["backend"] == "linear"
        rows = payload["rows"]
        assert isinstance(rows, list)
        assert len(rows) == 4
        assert rows[0]["cached_count"] >= 0
        assert isinstance(rows[0]["router_weights"], list)
        assert len(rows[0]["router_weights"]) >= 1


def test_bench_niah_model_backed_emits_model_backed_adapter_type(tmp_path) -> None:
    runner = CliRunner()
    checkpoint = _write_tiny_mc_checkpoint(tmp_path)
    result = runner.invoke(
        app,
        [
            "bench",
            "niah",
            "--adapter",
            "model",
            "--model-checkpoint",
            checkpoint,
            "--model-device",
            "cpu",
            "--tasks",
            "s_niah_1",
            "--context-lengths",
            "64",
            "--samples-per-length",
            "1",
            "--seed",
            "0",
            "--out-dir",
            str(tmp_path / "bench_out"),
        ],
    )

    assert result.exit_code == 0
    json_start = result.output.find("{")
    assert json_start >= 0
    payload = json.loads(result.output[json_start:])
    assert payload["adapter_type"] == "model_backed"
