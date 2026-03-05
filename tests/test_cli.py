from __future__ import annotations

import json

from typer.testing import CliRunner

from memory_caching.cli import app


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
