from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .backends.dla import DLABackend
from .backends.linear import LinearMemoryBackend
from .backends.swla import SWLABackend
from .backends.titans import TitansBackend
from .config import DLAConfig, MCConfig, SWLAConfig, TitansConfig
from .layer import MemoryCachingLayer


def _build_backend(config: MCConfig):
    if config.backend == "linear":
        return LinearMemoryBackend()
    if config.backend == "dla":
        return DLABackend(config.dla)
    if config.backend == "titans":
        return TitansBackend(config.titans)
    if config.backend == "swla":
        return SWLABackend(config.swla)
    raise ValueError(f"unsupported backend: {config.backend}")


class TinyLM(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.norm(x)
        return self.head(x)


class TinyMemoryCachingLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        backend: str,
        aggregation: str,
        segment_size: int,
        segmentation: str = "constant",
        state_init_mode: str = "checkpoint",
        ssc_top_k: int = 2,
        dla: DLAConfig | None = None,
        titans: TitansConfig | None = None,
        swla: SWLAConfig | None = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mc_config = MCConfig(
            d_model=d_model,
            num_heads=num_heads,
            backend=backend,
            aggregation=aggregation,
            segmentation=segmentation,
            segment_size=segment_size,
            state_init_mode=state_init_mode,
            ssc_top_k=ssc_top_k,
            dla=dla if dla is not None else DLAConfig(),
            titans=titans if titans is not None else TitansConfig(),
            swla=swla if swla is not None else SWLAConfig(),
        )
        self.mc = MemoryCachingLayer(
            config=self.mc_config,
            backend=_build_backend(self.mc_config),
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.mc(x)
        x = self.norm(x)
        return self.head(x)


def build_tiny_model_from_spec(spec: Mapping[str, Any]) -> nn.Module:
    model_family = str(spec.get("model_family", "tiny_lm")).strip().lower()
    vocab_size = int(spec.get("vocab_size", 256))
    d_model = int(spec.get("d_model", 64))
    if model_family == "tiny_lm":
        return TinyLM(vocab_size=vocab_size, d_model=d_model)
    if model_family != "tiny_mc_lm":
        raise ValueError(f"unsupported model_family: {model_family}")

    dla_cfg = spec.get("dla", {})
    titans_cfg = spec.get("titans", {})
    swla_cfg = spec.get("swla", {})
    return TinyMemoryCachingLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=int(spec.get("num_heads", 2)),
        backend=str(spec.get("backend", "linear")),
        aggregation=str(spec.get("aggregation", "grm")),
        segment_size=int(spec.get("segment_size", 16)),
        segmentation=str(spec.get("segmentation", "constant")),
        state_init_mode=str(spec.get("state_init_mode", "checkpoint")),
        ssc_top_k=int(spec.get("ssc_top_k", 2)),
        dla=DLAConfig(**dla_cfg) if isinstance(dla_cfg, dict) else DLAConfig(),
        titans=TitansConfig(**titans_cfg) if isinstance(titans_cfg, dict) else TitansConfig(),
        swla=SWLAConfig(**swla_cfg) if isinstance(swla_cfg, dict) else SWLAConfig(),
    )


def load_tiny_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected checkpoint payload dict")
    model_spec = payload.get("model_spec")
    if not isinstance(model_spec, dict):
        raise ValueError(f"{path}: missing model_spec in checkpoint payload")
    model = build_tiny_model_from_spec(model_spec)
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"{path}: missing model_state in checkpoint payload")
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, payload


@dataclass(frozen=True)
class ByteTokenizer:
    vocab_limit: int = 256

    def encode(self, text: str, *, max_tokens: int | None = None) -> list[int]:
        token_ids = [int(byte) % self.vocab_limit for byte in text.encode("utf-8", errors="ignore")]
        if max_tokens is not None and max_tokens > 0:
            token_ids = token_ids[-max_tokens:]
        if not token_ids:
            return [0]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        data = bytes(int(token_id) % self.vocab_limit for token_id in token_ids)
        return data.decode("utf-8", errors="ignore")


@torch.no_grad()
def greedy_generate_text(
    *,
    model: nn.Module,
    tokenizer: ByteTokenizer,
    prompt: str,
    device: str | torch.device,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    encoded = tokenizer.encode(prompt, max_tokens=max_input_tokens)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(max(1, max_new_tokens)):
        logits = model(input_ids)
        next_token = int(logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_token)
        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        if input_ids.shape[1] > max_input_tokens:
            input_ids = input_ids[:, -max_input_tokens:]
    return tokenizer.decode(generated).strip()
