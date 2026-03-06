from __future__ import annotations

from memory_caching.scientific_manifest import (
    build_train_manifest,
    validate_benchmark_manifest,
    validate_train_manifest,
)


def test_build_train_manifest_truthfulness_for_mc_model() -> None:
    manifest = build_train_manifest(
        model_family="tiny_mc_lm",
        uses_memory_caching=True,
        checkpoint_path="artifacts/checkpoints/target/step_000002.pt",
        tokenizer={"kind": "byte"},
        config_path="configs/train/target.yaml",
        backend="linear",
        aggregation="grm",
        seed=0,
        training_data={"source": "data/processed", "source_type": "token_stream_jsonl"},
        architecture={
            "model_family": "tiny_mc_lm",
            "d_model": 8,
            "vocab_size": 256,
            "backend": "linear",
            "aggregation": "grm",
            "num_heads": 2,
            "segment_size": 2,
        },
    )
    assert validate_train_manifest(manifest) == []


def test_validate_train_manifest_rejects_false_mc_flag() -> None:
    manifest = build_train_manifest(
        model_family="tiny_mc_lm",
        uses_memory_caching=False,
        checkpoint_path="artifacts/checkpoints/target/step_000002.pt",
        tokenizer={"kind": "byte"},
        config_path="configs/train/target.yaml",
        backend="linear",
        aggregation="grm",
        seed=0,
        training_data={"source": "data/processed", "source_type": "token_stream_jsonl"},
        architecture={
            "model_family": "tiny_mc_lm",
            "d_model": 8,
            "vocab_size": 256,
            "backend": "linear",
            "aggregation": "grm",
            "num_heads": 2,
            "segment_size": 2,
        },
    )
    errors = validate_train_manifest(manifest)
    assert any("uses_memory_caching=true" in err for err in errors)


def test_validate_benchmark_manifest_requires_model_info_for_model_backed() -> None:
    manifest = {"run_type": "niah", "adapter_type": "model_backed", "model_info": {}}
    metrics = {"adapter_type": "model_backed"}
    errors = validate_benchmark_manifest(manifest, metrics)
    assert any("model_info missing key" in err for err in errors)


def test_validate_benchmark_manifest_accepts_complete_model_backed_entry() -> None:
    manifest = {
        "run_type": "niah",
        "adapter_type": "model_backed",
        "model_info": {
            "model_family": "tiny_mc_lm",
            "checkpoint_path": "artifacts/checkpoints/target/step_000002.pt",
            "tokenizer_kind": "byte",
            "device": "cpu",
            "generation_mode": "greedy",
        },
    }
    metrics = {"adapter_type": "model_backed"}
    assert validate_benchmark_manifest(manifest, metrics) == []
