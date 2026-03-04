from __future__ import annotations

import json

import pytest

from memory_caching.bench.adapters import DLAMCAdapter, LinearMCAdapter, TitansMCAdapter
from memory_caching.bench.artifacts import create_bundle, write_artifacts
from memory_caching.bench.config import BenchmarkConfig
from memory_caching.bench.mqar import generate_mqar_examples
from memory_caching.bench.niah import generate_niah_examples, normalize_answer
from memory_caching.bench.runner import (
    get_runner,
    list_runners,
    run_longbench_suite,
    run_mqar_suite,
    run_niah_suite,
    run_retrieval_suite,
)


def test_benchmark_config_validation() -> None:
    cfg = BenchmarkConfig(
        runner="niah",
        task="s_niah_1",
        lengths=(4096, 8192),
        seed=0,
        adapter="all",
    )
    assert cfg.runner == "niah"


def test_niah_generation_is_deterministic() -> None:
    a = generate_niah_examples(task="s_niah_1", context_length=1024, samples=4, seed=7)
    b = generate_niah_examples(task="s_niah_1", context_length=1024, samples=4, seed=7)
    assert a == b


def test_mqar_generation_is_deterministic() -> None:
    a = generate_mqar_examples(samples=4, num_pairs=8, num_queries=2, seed=11)
    b = generate_mqar_examples(samples=4, num_pairs=8, num_queries=2, seed=11)
    assert a == b


def test_niah_normalization_policy() -> None:
    assert normalize_answer("  A   B  ") == "a b"


def test_runner_registry_and_lookup() -> None:
    runners = list_runners()
    assert set(["niah", "mqar", "longbench", "retrieval"]).issubset(set(runners))
    assert get_runner("niah") is not None


def test_run_niah_suite_with_three_adapters() -> None:
    result = run_niah_suite(
        adapters=[LinearMCAdapter(), DLAMCAdapter(), TitansMCAdapter()],
        tasks=["s_niah_1", "s_niah_2"],
        context_lengths=[1024, 2048],
        samples_per_length=3,
        seed=0,
        position_mode="uniform",
    )
    assert result["benchmark"] == "niah"
    assert len(result["rows"]) == 3 * 2 * 2


def test_run_mqar_suite_with_three_adapters_has_micro_macro() -> None:
    result = run_mqar_suite(
        adapters=[LinearMCAdapter(), DLAMCAdapter(), TitansMCAdapter()],
        samples=6,
        num_pairs=8,
        num_queries=2,
        seed=0,
    )
    assert result["benchmark"] == "mqar"
    assert len(result["rows"]) == 3
    assert "micro_accuracy" in result["rows"][0]
    assert "macro_accuracy" in result["rows"][0]


def test_run_longbench_suite_scaffold() -> None:
    result = run_longbench_suite(
        adapters=[LinearMCAdapter(), TitansMCAdapter()],
        tasks=["single_doc_qa", "code"],
        samples_per_task=2,
        seed=0,
    )
    assert result["benchmark"] == "longbench"
    assert len(result["rows"]) == 4


def test_run_longbench_suite_with_dataset_file(tmp_path) -> None:
    dataset_file = tmp_path / "longbench.jsonl"
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_group": "single_doc_qa",
                        "prompt": "QUESTION: reply with ANSWER_OK\nANSWER:",
                        "answer": "ANSWER_OK",
                    }
                ),
                json.dumps(
                    {
                        "task_group": "single_doc_qa",
                        "prompt": "QUESTION: reply with ANSWER_OK\nANSWER:",
                        "answer": "ANSWER_OK",
                    }
                ),
            ]
        )
        + "\n"
    )
    result = run_longbench_suite(
        adapters=[LinearMCAdapter()],
        tasks=["single_doc_qa"],
        samples_per_task=2,
        seed=0,
        dataset_file=str(dataset_file),
    )
    assert result["benchmark"] == "longbench"
    assert len(result["rows"]) == 1


def test_run_longbench_unknown_task_raises() -> None:
    with pytest.raises(ValueError):
        run_longbench_suite(
            adapters=[LinearMCAdapter()],
            tasks=["unknown_task"],
            samples_per_task=1,
            seed=0,
        )


def test_run_retrieval_suite_and_truncation_contract() -> None:
    result = run_retrieval_suite(
        adapters=[LinearMCAdapter(), DLAMCAdapter()],
        datasets=["swde", "squad"],
        truncation_lengths=[512, 1024],
        samples_per_dataset=2,
        seed=0,
    )
    assert result["benchmark"] == "retrieval"
    assert len(result["rows"]) == 8


def test_run_retrieval_suite_with_dataset_file(tmp_path) -> None:
    dataset_file = tmp_path / "retrieval.jsonl"
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "swde",
                        "document": "retrieval evidence row 1",
                        "question": "return RETRIEVAL_OK",
                        "answer": "RETRIEVAL_OK",
                    }
                ),
                json.dumps(
                    {
                        "dataset": "swde",
                        "document": "retrieval evidence row 2",
                        "question": "return RETRIEVAL_OK",
                        "answer": "RETRIEVAL_OK",
                    }
                ),
            ]
        )
        + "\n"
    )
    result = run_retrieval_suite(
        adapters=[LinearMCAdapter()],
        datasets=["swde"],
        truncation_lengths=[64],
        samples_per_dataset=2,
        seed=0,
        dataset_file=str(dataset_file),
    )
    assert result["benchmark"] == "retrieval"
    assert len(result["rows"]) == 1


@pytest.mark.parametrize("bad_len", [0, -1])
def test_retrieval_invalid_truncation_raises(bad_len: int) -> None:
    with pytest.raises(ValueError):
        run_retrieval_suite(
            adapters=[LinearMCAdapter()],
            datasets=["swde"],
            truncation_lengths=[bad_len],
            samples_per_dataset=1,
            seed=0,
        )


def test_retrieval_invalid_dataset_raises() -> None:
    with pytest.raises(ValueError):
        run_retrieval_suite(
            adapters=[LinearMCAdapter()],
            datasets=["bad_dataset"],
            truncation_lengths=[512],
            samples_per_dataset=1,
            seed=0,
        )


def test_artifact_bundle_writes_manifest_metrics_rows_csv_report(tmp_path) -> None:
    bundle = create_bundle(str(tmp_path))
    metrics = {
        "benchmark": "niah",
        "mean_accuracy": 1.0,
        "rows": [{"adapter": "linear-mc", "accuracy": 1.0}],
    }
    cfg = {"adapter": "all", "samples_per_length": 4}

    write_artifacts(
        bundle=bundle,
        run_type="niah",
        config=cfg,
        metrics=metrics,
        runner_version="v0.2",
        dataset_revision="synthetic-v2",
    )

    on_disk_metrics = json.loads(bundle.metrics_path.read_text())
    manifest = json.loads(bundle.manifest_path.read_text())

    assert on_disk_metrics == metrics
    assert manifest["schema_version"] == "v1"
    assert manifest["run_type"] == "niah"
    assert manifest["config"] == cfg
    assert manifest["metrics_file"] == str(bundle.metrics_path)
    assert manifest["runner_version"] == "v0.2"
    assert manifest["dataset_revision"] == "synthetic-v2"
    assert bundle.rows_path.exists()
    assert bundle.summary_csv_path.exists()
    assert bundle.report_md_path.exists()
