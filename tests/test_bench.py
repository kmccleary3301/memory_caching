from __future__ import annotations

import json

from memory_caching.bench.adapters import DLAMCAdapter, LinearMCAdapter
from memory_caching.bench.artifacts import create_bundle, write_artifacts
from memory_caching.bench.mqar import generate_mqar_examples
from memory_caching.bench.niah import generate_niah_examples
from memory_caching.bench.runner import run_mqar_suite, run_niah_suite


def test_niah_generation_is_deterministic() -> None:
    a = generate_niah_examples(task="s_niah_1", context_length=1024, samples=4, seed=7)
    b = generate_niah_examples(task="s_niah_1", context_length=1024, samples=4, seed=7)
    assert a == b


def test_mqar_generation_is_deterministic() -> None:
    a = generate_mqar_examples(samples=4, num_pairs=8, num_queries=2, seed=11)
    b = generate_mqar_examples(samples=4, num_pairs=8, num_queries=2, seed=11)
    assert a == b


def test_run_niah_suite_with_two_adapters() -> None:
    result = run_niah_suite(
        adapters=[LinearMCAdapter(), DLAMCAdapter()],
        tasks=["s_niah_1", "s_niah_2"],
        context_lengths=[1024, 2048],
        samples_per_length=3,
        seed=0,
    )
    assert result["benchmark"] == "niah"
    assert len(result["rows"]) == 2 * 2 * 2


def test_run_mqar_suite_with_two_adapters() -> None:
    result = run_mqar_suite(
        adapters=[LinearMCAdapter(), DLAMCAdapter()],
        samples=6,
        num_pairs=8,
        num_queries=2,
        seed=0,
    )
    assert result["benchmark"] == "mqar"
    assert len(result["rows"]) == 2


def test_artifact_bundle_writes_manifest_and_metrics(tmp_path) -> None:
    bundle = create_bundle(str(tmp_path))
    metrics = {"benchmark": "niah", "mean_accuracy": 1.0, "rows": []}
    cfg = {"adapter": "both", "samples_per_length": 4}

    write_artifacts(bundle=bundle, run_type="niah", config=cfg, metrics=metrics)

    on_disk_metrics = json.loads(bundle.metrics_path.read_text())
    manifest = json.loads(bundle.manifest_path.read_text())

    assert on_disk_metrics == metrics
    assert manifest["schema_version"] == "v1"
    assert manifest["run_type"] == "niah"
    assert manifest["config"] == cfg
    assert manifest["metrics_file"] == str(bundle.metrics_path)
