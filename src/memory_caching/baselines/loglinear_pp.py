from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from torch import Tensor, nn

from ..backends.dla import DLABackend
from ..backends.linear import LinearMemoryBackend
from ..backends.swla import SWLABackend
from ..backends.titans import TitansBackend
from ..config import MCConfig
from ..layer import MemoryCachingLayer


@dataclass(frozen=True)
class LogLinearPPConfig:
    """Explicit preset for the Memory Caching paper's Log-Linear++ baseline.

    This preset is intentionally defined as:

    - aggregation = "grm"
    - segmentation = "logarithmic"

    over the existing MemoryCachingLayer abstraction.
    """

    d_model: int
    num_heads: int
    backend: str = "linear"
    state_init_mode: str = "checkpoint"
    mc_config_overrides: Mapping[str, Any] = field(default_factory=dict)

    def build_mc_config(self) -> MCConfig:
        overrides = dict(self.mc_config_overrides)

        forbidden = {
            "aggregation": "grm",
            "segmentation": "logarithmic",
        }
        for key, expected in forbidden.items():
            if key in overrides and overrides[key] != expected:
                raise ValueError(
                    f"LogLinearPP requires {key}={expected!r}; received {overrides[key]!r}"
                )

        overrides.setdefault("d_model", self.d_model)
        overrides.setdefault("num_heads", self.num_heads)
        overrides.setdefault("backend", self.backend)
        overrides.setdefault("aggregation", "grm")
        overrides.setdefault("segmentation", "logarithmic")
        overrides.setdefault("state_init_mode", self.state_init_mode)

        return MCConfig(**overrides)


class LogLinearPP(nn.Module):
    """Memory Caching paper baseline preset for Log-Linear++.

    This module is deliberately a constrained wrapper around MemoryCachingLayer.
    It is not the original Guo et al. Log-Linear Attention mechanism.
    """

    def __init__(self, config: LogLinearPPConfig) -> None:
        super().__init__()
        self.config = config
        self.mc_config = config.build_mc_config()
        self.layer = MemoryCachingLayer(
            config=self.mc_config,
            backend=_build_backend(self.mc_config),
        )

    def forward(self, x: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        return self.layer(x, attention_mask=attention_mask)

    def forward_with_cache(
        self, x: Tensor, *, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, list[Any]]:
        return self.layer.forward_with_cache(x, attention_mask=attention_mask)

    def inspect(
        self, x: Tensor, *, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, list[dict[str, Any]]]:
        return self.layer.inspect(x, attention_mask=attention_mask)


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
