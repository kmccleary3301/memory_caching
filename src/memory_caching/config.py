from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AggregationVariant = Literal["residual", "grm", "soup", "ssc"]
SegmentationMode = Literal["constant", "logarithmic"]
StateInitMode = Literal["checkpoint", "restart"]
BackendKind = Literal["linear", "dla"]
DLAObjective = Literal["dot", "l2"]
DLAInnerUpdateMode = Literal["stopgrad", "differentiable"]


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
class MCConfig:
    d_model: int
    num_heads: int
    backend: BackendKind = "linear"
    dla: DLAConfig = field(default_factory=DLAConfig)
    aggregation: AggregationVariant = "grm"
    segmentation: SegmentationMode = "constant"
    segment_size: int = 256
    state_init_mode: StateInitMode = "checkpoint"
    ssc_top_k: int = 2
    use_q_as_u: bool = False
    softmax_temperature: float = 1.0
    detach_cached_states: bool = False
    detach_cached_context: bool = False

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
