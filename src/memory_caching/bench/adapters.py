from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..models import ByteTokenizer, greedy_generate_text, load_tiny_model_checkpoint


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str
    predictor: Callable[[str], str]

    def predict(self, prompt: str) -> str:
        return self.predictor(prompt)


@dataclass(frozen=True)
class ModelBackedAdapter(BenchmarkAdapter):
    backend_kind: str
    checkpoint_path: str | None = None
    metadata: Mapping[str, Any] | None = None


_PASSKEY_RE = re.compile(r"PASSKEY\s*:\s*([A-Za-z0-9\-]+)")
_NUMBER_RE = re.compile(r"NEEDLE_NUMBER\s*:\s*([0-9]+)")
_UUID_RE = re.compile(r"NEEDLE_UUID\s*:\s*([0-9a-fA-F\-]+)")
_MQAR_RE = re.compile(r"QUERY\s*:\s*([A-Z0-9_]+)")
_PAIR_RE = re.compile(r"PAIR\s+([A-Z0-9_]+)\s*->\s*([A-Z0-9_]+)")


def _predict_generic(prompt: str) -> str:
    for rx in (_PASSKEY_RE, _NUMBER_RE, _UUID_RE):
        m = rx.search(prompt)
        if m:
            return m.group(1).strip()

    query = _MQAR_RE.search(prompt)
    if query is not None:
        target = query.group(1).strip()
        pairs = dict(_PAIR_RE.findall(prompt))
        if target in pairs:
            return pairs[target]

    if "ANSWER_OK" in prompt:
        return "ANSWER_OK"
    if "RETRIEVAL_OK" in prompt:
        return "RETRIEVAL_OK"
    return ""


class LinearMCAdapter(BenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="linear-mc",
            predictor=_predict_generic,
        )


class DLAMCAdapter(BenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="dla-mc",
            predictor=_predict_generic,
        )


class TitansMCAdapter(BenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="titans-mc",
            predictor=_predict_generic,
        )


def make_model_backed_adapter(
    *,
    name: str,
    backend_kind: str,
    predictor: Callable[[str], str],
    checkpoint_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModelBackedAdapter:
    return ModelBackedAdapter(
        name=name,
        predictor=predictor,
        backend_kind=backend_kind,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )


def make_checkpoint_model_backed_adapter(
    *,
    checkpoint_path: str,
    device: str = "cpu",
    max_new_tokens: int = 16,
    max_input_tokens: int = 512,
    seed: int = 0,
    name: str | None = None,
) -> ModelBackedAdapter:
    resolved = Path(checkpoint_path).resolve()
    model, payload = load_tiny_model_checkpoint(resolved, device=device)
    tokenizer = ByteTokenizer()
    model_spec = payload.get("model_spec", {})
    if not isinstance(model_spec, dict):
        raise ValueError(f"{resolved}: checkpoint missing model_spec")
    backend_kind = str(model_spec.get("backend", model_spec.get("model_family", "unknown")))

    def _predict(prompt: str) -> str:
        return greedy_generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

    metadata: dict[str, Any] = {
        "model_family": str(model_spec.get("model_family", "unknown")),
        "checkpoint_path": str(resolved),
        "tokenizer_kind": "byte",
        "device": str(device),
        "generation_mode": "greedy",
        "max_new_tokens": int(max_new_tokens),
        "max_input_tokens": int(max_input_tokens),
        "seed": int(seed),
        "backend": str(model_spec.get("backend", "")),
        "aggregation": str(model_spec.get("aggregation", "")),
    }
    return make_model_backed_adapter(
        name=name or f"checkpoint-model::{resolved.stem}",
        backend_kind=backend_kind,
        predictor=_predict,
        checkpoint_path=str(resolved),
        metadata=metadata,
    )
