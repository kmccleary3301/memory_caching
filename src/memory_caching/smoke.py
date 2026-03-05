from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends.dla import DLABackend
from .backends.linear import LinearMemoryBackend
from .backends.swla import SWLABackend
from .backends.titans import TitansBackend
from .config import BackendKind, DLAConfig, MCConfig, SWLAConfig, TitansConfig
from .layer import MemoryCachingLayer

Aggregation = Literal["residual", "grm", "soup", "ssc"]
Segmentation = Literal["constant", "logarithmic"]
StateInit = Literal["checkpoint", "restart"]

_VALID_AGGREGATIONS = {"residual", "grm", "soup", "ssc"}
_VALID_SEGMENTATIONS = {"constant", "logarithmic"}
_VALID_STATE_INIT = {"checkpoint", "restart"}
_VALID_BACKENDS = {"linear", "dla", "titans", "swla"}
_VALID_TITANS_UPDATE_CONVENTIONS = {"paper", "gradient_descent"}


@dataclass(frozen=True)
class SmokeMetrics:
    mode: str
    device: str
    backend: str
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
        self.mc = MemoryCachingLayer(config=config, backend=_build_backend(config))
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(input_ids)
        x = self.mc(x)
        x = self.norm(x)
        return self.lm_head(x)


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


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _validate_modes(
    *,
    backend: str,
    aggregation: str,
    segmentation: str,
    state_init_mode: str,
) -> tuple[BackendKind, Aggregation, Segmentation, StateInit]:
    backend_kind = backend.strip().lower()
    agg = aggregation.strip().lower()
    seg = segmentation.strip().lower()
    init_mode = state_init_mode.strip().lower()

    if backend_kind not in _VALID_BACKENDS:
        raise ValueError(f"invalid backend: {backend}")
    if agg not in _VALID_AGGREGATIONS:
        raise ValueError(f"invalid aggregation: {aggregation}")
    if seg not in _VALID_SEGMENTATIONS:
        raise ValueError(f"invalid segmentation: {segmentation}")
    if init_mode not in _VALID_STATE_INIT:
        raise ValueError(f"invalid state_init_mode: {state_init_mode}")

    return backend_kind, agg, seg, init_mode


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


def _write_metrics(metrics: dict[str, float | int | str], out_json: str | None) -> None:
    if out_json is None:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n")


def _build_config(
    *,
    d_model: int,
    num_heads: int,
    backend: str,
    dla_memory_width: int,
    dla_memory_depth: int,
    dla_objective: str,
    dla_inner_update_mode: str,
    dla_step_size: float,
    dla_momentum: float,
    titans_memory_width: int,
    titans_memory_depth: int,
    titans_objective: str,
    titans_inner_update_mode: str,
    titans_step_size: float,
    titans_momentum: float,
    titans_retention_alpha: float,
    titans_update_convention: str,
    swla_alpha: float,
    swla_beta: float,
    swla_lam: float,
    aggregation: str,
    segmentation: str,
    segment_size: int,
    state_init_mode: str,
    ssc_top_k: int,
) -> MCConfig:
    if titans_update_convention not in _VALID_TITANS_UPDATE_CONVENTIONS:
        raise ValueError(
            "invalid titans_update_convention: "
            f"{titans_update_convention} (expected one of {_VALID_TITANS_UPDATE_CONVENTIONS})"
        )
    return MCConfig(
        d_model=d_model,
        num_heads=num_heads,
        backend=backend,
        dla=DLAConfig(
            memory_width=dla_memory_width,
            memory_depth=dla_memory_depth,
            objective=dla_objective,
            inner_update_mode=dla_inner_update_mode,
            step_size=dla_step_size,
            momentum=dla_momentum,
        ),
        titans=TitansConfig(
            memory_width=titans_memory_width,
            memory_depth=titans_memory_depth,
            objective=titans_objective,
            inner_update_mode=titans_inner_update_mode,
            step_size=titans_step_size,
            momentum=titans_momentum,
            retention_alpha=titans_retention_alpha,
            update_convention=titans_update_convention,
        ),
        swla=SWLAConfig(alpha=swla_alpha, beta=swla_beta, lam=swla_lam),
        aggregation=aggregation,
        segmentation=segmentation,
        segment_size=segment_size,
        state_init_mode=state_init_mode,
        ssc_top_k=ssc_top_k,
    )


def run_smoke_train(
    *,
    steps: int = 20,
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 128,
    d_model: int = 64,
    num_heads: int = 4,
    backend: str = "linear",
    dla_memory_width: int = 64,
    dla_memory_depth: int = 2,
    dla_objective: str = "dot",
    dla_inner_update_mode: str = "stopgrad",
    dla_step_size: float = 0.05,
    dla_momentum: float = 0.0,
    titans_memory_width: int = 64,
    titans_memory_depth: int = 2,
    titans_objective: str = "l2",
    titans_inner_update_mode: str = "stopgrad",
    titans_step_size: float = 0.05,
    titans_momentum: float = 0.9,
    titans_retention_alpha: float = 1.0,
    titans_update_convention: str = "paper",
    swla_alpha: float = 1.0,
    swla_beta: float = 0.0,
    swla_lam: float = 1.0,
    segment_size: int = 16,
    aggregation: str = "grm",
    segmentation: str = "constant",
    state_init_mode: str = "checkpoint",
    ssc_top_k: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    out_json: str | None = None,
) -> dict[str, float | int | str]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")

    backend_kind, agg, seg, init_mode = _validate_modes(
        backend=backend,
        aggregation=aggregation,
        segmentation=segmentation,
        state_init_mode=state_init_mode,
    )

    torch.manual_seed(seed)
    resolved_device = _resolve_device(device)

    config = _build_config(
        d_model=d_model,
        num_heads=num_heads,
        backend=backend_kind,
        dla_memory_width=dla_memory_width,
        dla_memory_depth=dla_memory_depth,
        dla_objective=dla_objective,
        dla_inner_update_mode=dla_inner_update_mode,
        dla_step_size=dla_step_size,
        dla_momentum=dla_momentum,
        titans_memory_width=titans_memory_width,
        titans_memory_depth=titans_memory_depth,
        titans_objective=titans_objective,
        titans_inner_update_mode=titans_inner_update_mode,
        titans_step_size=titans_step_size,
        titans_momentum=titans_momentum,
        titans_retention_alpha=titans_retention_alpha,
        titans_update_convention=titans_update_convention,
        swla_alpha=swla_alpha,
        swla_beta=swla_beta,
        swla_lam=swla_lam,
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
        backend=backend_kind,
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
    ).to_dict()
    _write_metrics(metrics, out_json)
    return metrics


def run_smoke_eval(
    *,
    warmup_steps: int = 0,
    batch_size: int = 4,
    seq_len: int = 64,
    vocab_size: int = 128,
    d_model: int = 64,
    num_heads: int = 4,
    backend: str = "linear",
    dla_memory_width: int = 64,
    dla_memory_depth: int = 2,
    dla_objective: str = "dot",
    dla_inner_update_mode: str = "stopgrad",
    dla_step_size: float = 0.05,
    dla_momentum: float = 0.0,
    titans_memory_width: int = 64,
    titans_memory_depth: int = 2,
    titans_objective: str = "l2",
    titans_inner_update_mode: str = "stopgrad",
    titans_step_size: float = 0.05,
    titans_momentum: float = 0.9,
    titans_retention_alpha: float = 1.0,
    titans_update_convention: str = "paper",
    swla_alpha: float = 1.0,
    swla_beta: float = 0.0,
    swla_lam: float = 1.0,
    segment_size: int = 16,
    aggregation: str = "grm",
    segmentation: str = "constant",
    state_init_mode: str = "checkpoint",
    ssc_top_k: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    out_json: str | None = None,
) -> dict[str, float | int | str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    backend_kind, agg, seg, init_mode = _validate_modes(
        backend=backend,
        aggregation=aggregation,
        segmentation=segmentation,
        state_init_mode=state_init_mode,
    )

    torch.manual_seed(seed)
    resolved_device = _resolve_device(device)

    config = _build_config(
        d_model=d_model,
        num_heads=num_heads,
        backend=backend_kind,
        dla_memory_width=dla_memory_width,
        dla_memory_depth=dla_memory_depth,
        dla_objective=dla_objective,
        dla_inner_update_mode=dla_inner_update_mode,
        dla_step_size=dla_step_size,
        dla_momentum=dla_momentum,
        titans_memory_width=titans_memory_width,
        titans_memory_depth=titans_memory_depth,
        titans_objective=titans_objective,
        titans_inner_update_mode=titans_inner_update_mode,
        titans_step_size=titans_step_size,
        titans_momentum=titans_momentum,
        titans_retention_alpha=titans_retention_alpha,
        titans_update_convention=titans_update_convention,
        swla_alpha=swla_alpha,
        swla_beta=swla_beta,
        swla_lam=swla_lam,
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
        backend=backend_kind,
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
    ).to_dict()
    _write_metrics(metrics, out_json)
    return metrics
