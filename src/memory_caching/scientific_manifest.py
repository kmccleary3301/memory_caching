from __future__ import annotations

from typing import Any, Mapping


def build_train_manifest(
    *,
    model_family: str,
    uses_memory_caching: bool,
    checkpoint_path: str,
    tokenizer: Mapping[str, Any],
    config_path: str,
    backend: str | None,
    aggregation: str | None,
    seed: int,
    training_data: Mapping[str, Any],
    architecture: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "scientific_train_manifest_v1",
        "model_family": model_family,
        "uses_memory_caching": uses_memory_caching,
        "checkpoint_path": checkpoint_path,
        "tokenizer": dict(tokenizer),
        "config_path": config_path,
        "backend": backend,
        "aggregation": aggregation,
        "seed": int(seed),
        "training_data": dict(training_data),
        "architecture": dict(architecture),
    }
    if extra:
        payload.update(extra)
    return payload


def validate_train_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "schema_version",
        "model_family",
        "uses_memory_caching",
        "checkpoint_path",
        "tokenizer",
        "config_path",
        "seed",
        "training_data",
        "architecture",
    ]
    for key in required:
        if key not in manifest:
            errors.append(f"train manifest missing key: {key}")

    model_family = str(manifest.get("model_family", "")).strip().lower()
    if model_family not in {"tiny_lm", "tiny_mc_lm"}:
        errors.append(f"unsupported model_family: {model_family or 'missing'}")

    uses_memory_caching = manifest.get("uses_memory_caching")
    if not isinstance(uses_memory_caching, bool):
        errors.append("uses_memory_caching must be a boolean")
    elif model_family == "tiny_mc_lm" and uses_memory_caching is not True:
        errors.append("tiny_mc_lm manifests must record uses_memory_caching=true")
    elif model_family == "tiny_lm" and uses_memory_caching is not False:
        errors.append("tiny_lm manifests must record uses_memory_caching=false")

    tokenizer = manifest.get("tokenizer", {})
    if not isinstance(tokenizer, Mapping):
        errors.append("tokenizer must be an object")
    else:
        kind = str(tokenizer.get("kind", "")).strip().lower()
        if not kind:
            errors.append("tokenizer.kind must be present")

    training_data = manifest.get("training_data", {})
    if not isinstance(training_data, Mapping):
        errors.append("training_data must be an object")
    else:
        if not str(training_data.get("source", "")).strip():
            errors.append("training_data.source must be present")
        if not str(training_data.get("source_type", "")).strip():
            errors.append("training_data.source_type must be present")

    architecture = manifest.get("architecture", {})
    if not isinstance(architecture, Mapping):
        errors.append("architecture must be an object")
    else:
        for key in ("model_family", "d_model", "vocab_size"):
            if key not in architecture:
                errors.append(f"architecture missing key: {key}")
        if uses_memory_caching is True:
            for key in ("backend", "aggregation", "num_heads", "segment_size"):
                if key not in architecture:
                    errors.append(f"MC architecture missing key: {key}")

    if uses_memory_caching is True:
        if not str(manifest.get("backend", "")).strip():
            errors.append("MC train manifests must record backend")
        if not str(manifest.get("aggregation", "")).strip():
            errors.append("MC train manifests must record aggregation")

    return errors


def validate_benchmark_manifest(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    run_type = str(manifest.get("run_type", "")).strip().lower()
    if run_type not in {"niah", "mqar", "longbench", "retrieval"}:
        errors.append(f"unsupported benchmark run_type: {run_type or 'missing'}")

    adapter_type = str(
        manifest.get(
            "adapter_type",
            metrics.get("adapter_type", manifest.get("config", {}).get("adapter_type", "")),
        )
    ).strip().lower()
    if adapter_type not in {"rule_based", "model_backed"}:
        errors.append(f"unsupported adapter_type: {adapter_type or 'missing'}")

    if adapter_type == "model_backed":
        model_info = manifest.get("model_info", manifest.get("config", {}).get("model_info", {}))
        if not isinstance(model_info, Mapping):
            errors.append("model_backed benchmark manifests must include model_info object")
        else:
            for key in ("model_family", "checkpoint_path", "tokenizer_kind", "device", "generation_mode"):
                if not str(model_info.get(key, "")).strip():
                    errors.append(f"model_info missing key: {key}")

    return errors
