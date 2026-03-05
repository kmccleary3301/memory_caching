from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

AggregationVariant = Literal["residual", "grm", "soup", "ssc"]
SegmentationMode = Literal["constant", "logarithmic"]
StateInitMode = Literal["checkpoint", "restart"]
BackendKind = Literal["linear", "dla", "titans", "swla"]
DLAObjective = Literal["dot", "l2"]
DLAInnerUpdateMode = Literal["stopgrad", "differentiable"]
TitansObjective = Literal["l2", "dot"]
TitansInnerUpdateMode = Literal["stopgrad", "differentiable"]
# `paper` tracks the written recursion form; `gradient_descent` keeps explicit
# descent-sign behavior for optimizer-style updates.
TitansUpdateConvention = Literal["paper", "gradient_descent"]


@dataclass(frozen=True)
class DLAConfig:
    memory_width: int = 64
    memory_depth: int = 2
    objective: DLAObjective = "dot"
    inner_update_mode: DLAInnerUpdateMode = "stopgrad"
    step_size: float = 0.05
    momentum: float = 0.0

    def __post_init__(self) -> None:
        if self.memory_width <= 0:
            raise ValueError("memory_width must be positive")
        if self.memory_depth < 2:
            raise ValueError("memory_depth must be at least 2")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")


@dataclass(frozen=True)
class TitansConfig:
    memory_width: int = 64
    memory_depth: int = 2
    objective: TitansObjective = "l2"
    inner_update_mode: TitansInnerUpdateMode = "stopgrad"
    step_size: float = 0.05
    momentum: float = 0.9
    retention_alpha: float = 1.0
    # `paper`: S_t = beta * S_{t-1} - eta * grad_t
    # `gradient_descent`: S_t = beta * S_{t-1} + eta * grad_t
    update_convention: TitansUpdateConvention = "paper"

    def __post_init__(self) -> None:
        if self.memory_width <= 0:
            raise ValueError("memory_width must be positive")
        if self.memory_depth < 2:
            raise ValueError("memory_depth must be at least 2")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if not (0.0 < self.retention_alpha <= 1.0):
            raise ValueError("retention_alpha must be in (0, 1]")
        if self.update_convention not in {"paper", "gradient_descent"}:
            raise ValueError("update_convention must be one of paper, gradient_descent")


@dataclass(frozen=True)
class SWLAConfig:
    alpha: float = 1.0
    beta: float = 0.0
    lam: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        if not isfinite(self.beta):
            raise ValueError("beta must be finite")
        if not isfinite(self.lam):
            raise ValueError("lam must be finite")


@dataclass(frozen=True)
class MCConfig:
    d_model: int
    num_heads: int
    backend: BackendKind = "linear"
    dla: DLAConfig = field(default_factory=DLAConfig)
    titans: TitansConfig = field(default_factory=TitansConfig)
    swla: SWLAConfig = field(default_factory=SWLAConfig)
    aggregation: AggregationVariant = "grm"
    segmentation: SegmentationMode = "constant"
    segment_size: int = 256
    state_init_mode: StateInitMode = "checkpoint"
    ssc_top_k: int = 2
    use_q_as_u: bool = False
    softmax_temperature: float = 1.0
    detach_cached_states: bool = False
    detach_cached_context: bool = False
    allow_output_mixture_fallback: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.segment_size <= 0:
            raise ValueError("segment_size must be positive")
        if self.ssc_top_k <= 0:
            raise ValueError("ssc_top_k must be positive")
        if self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        if self.backend not in {"linear", "dla", "titans", "swla"}:
            raise ValueError("backend must be one of linear, dla, titans, swla")
