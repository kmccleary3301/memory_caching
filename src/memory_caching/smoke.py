from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends.linear import LinearMemoryBackend
from .config import MCConfig
from .layer import MemoryCachingLayer

Aggregation = Literal["residual", "grm", "soup", "ssc"]
Segmentation = Literal["constant", "logarithmic"]
StateInit = Literal["checkpoint", "restart"]

_VALID_AGGREGATIONS = {"residual", "grm", "soup", "ssc"}
_VALID_SEGMENTATIONS = {"constant", "logarithmic"}
_VALID_STATE_INIT = {"checkpoint", "restart"}


@dataclass(frozen=True)
class SmokeMetrics:
    mode: str
    device: str
    steps: int
    batch_size: int
    seq_len: int
    vocab_size: int
    initial_loss: float
    final_loss: float
    eval_loss: float
    eval_accuracy: float
    cache_segments: int
    mean_segment_len: float
    trainable_params: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class TinyMCLanguageModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: MCConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

        self.token_embed = nn.Embedding(vocab_size, config.d_model)
        self.mc = MemoryCachingLayer(config=config, backend=LinearMemoryBackend())
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(input_ids)
        x = self.mc(x)
        x = self.norm(x)
        return self.lm_head(x)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _validate_modes(
    *, aggregation: str, segmentation: str, state_init_mode: str
) -> tuple[Aggregation, Segmentation, StateInit]:
    agg = aggregation.strip().lower()
    seg = segmentation.strip().lower()
    init_mode = state_init_mode.strip().lower()

    if agg not in _VALID_AGGREGATIONS:
        raise ValueError(f"invalid aggregation: {aggregation}")
    if seg not in _VALID_SEGMENTATIONS:
        raise ValueError(f"invalid segmentation: {segmentation}")
    if init_mode not in _VALID_STATE_INIT:
        raise ValueError(f"invalid state_init_mode: {state_init_mode}")

    return agg, seg, init_mode


def _make_batch(
    *, batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )


def _next_token_loss_and_accuracy(
    model: TinyMCLanguageModel,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits = model(inputs)

    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean().item()
    return loss, float(accuracy)


def _cache_stats(model: TinyMCLanguageModel, tokens: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        embeds = model.token_embed(tokens)
        _, cache = model.mc(embeds, return_cache=True)

    cache_segments = len(cache)
    mean_len = 0.0
    if cache_segments > 0:
        mean_len = float(sum(c.seg_len for c in cache) / cache_segments)
    return cache_segments, mean_len


def run_smoke_train(
    *,
    steps: int = 20,
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 128,
    d_model: int = 64,
    num_heads: int = 4,
    segment_size: int = 16,
    aggregation: str = "grm",
    segmentation: str = "constant",
    state_init_mode: str = "checkpoint",
    ssc_top_k: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
) -> dict[str, float | int | str]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")

    agg, seg, init_mode = _validate_modes(
        aggregation=aggregation,
        segmentation=segmentation,
        state_init_mode=state_init_mode,
    )

    torch.manual_seed(seed)
    resolved_device = _resolve_device(device)

    config = MCConfig(
        d_model=d_model,
        num_heads=num_heads,
        aggregation=agg,
        segmentation=seg,
        segment_size=segment_size,
        state_init_mode=init_mode,
        ssc_top_k=ssc_top_k,
    )

    model = TinyMCLanguageModel(vocab_size=vocab_size, config=config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    first_loss: float | None = None
    last_loss = 0.0

    for _ in range(steps):
        tokens = _make_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=resolved_device,
        )
        loss, _ = _next_token_loss_and_accuracy(model, tokens)

        if first_loss is None:
            first_loss = float(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())

    model.eval()
    eval_tokens = _make_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=resolved_device,
    )
    with torch.no_grad():
        eval_loss_tensor, eval_acc = _next_token_loss_and_accuracy(model, eval_tokens)
        eval_loss = float(eval_loss_tensor.item())

    cache_segments, mean_segment_len = _cache_stats(model, eval_tokens)

    metrics = SmokeMetrics(
        mode="train",
        device=str(resolved_device),
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        initial_loss=float(first_loss if first_loss is not None else last_loss),
        final_loss=last_loss,
        eval_loss=eval_loss,
        eval_accuracy=eval_acc,
        cache_segments=cache_segments,
        mean_segment_len=mean_segment_len,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return metrics.to_dict()


def run_smoke_eval(
    *,
    warmup_steps: int = 0,
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 128,
    d_model: int = 64,
    num_heads: int = 4,
    segment_size: int = 16,
    aggregation: str = "grm",
    segmentation: str = "constant",
    state_init_mode: str = "checkpoint",
    ssc_top_k: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
) -> dict[str, float | int | str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    agg, seg, init_mode = _validate_modes(
        aggregation=aggregation,
        segmentation=segmentation,
        state_init_mode=state_init_mode,
    )

    torch.manual_seed(seed)
    resolved_device = _resolve_device(device)

    config = MCConfig(
        d_model=d_model,
        num_heads=num_heads,
        aggregation=agg,
        segmentation=seg,
        segment_size=segment_size,
        state_init_mode=init_mode,
        ssc_top_k=ssc_top_k,
    )

    model = TinyMCLanguageModel(vocab_size=vocab_size, config=config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(warmup_steps):
        tokens = _make_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=resolved_device,
        )
        loss, _ = _next_token_loss_and_accuracy(model, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    eval_tokens = _make_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=resolved_device,
    )
    with torch.no_grad():
        eval_loss_tensor, eval_acc = _next_token_loss_and_accuracy(model, eval_tokens)
        eval_loss = float(eval_loss_tensor.item())

    cache_segments, mean_segment_len = _cache_stats(model, eval_tokens)

    metrics = SmokeMetrics(
        mode="eval",
        device=str(resolved_device),
        steps=warmup_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        initial_loss=eval_loss,
        final_loss=eval_loss,
        eval_loss=eval_loss,
        eval_accuracy=eval_acc,
        cache_segments=cache_segments,
        mean_segment_len=mean_segment_len,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return metrics.to_dict()
