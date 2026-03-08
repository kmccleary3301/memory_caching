"""Reference Log-Linear Attention implementation namespace.

This namespace is reserved for the original Guo et al. mechanism family.
It is intentionally separate from the Memory Caching paper's LogLinearPP
baseline preset.
"""

from .dense_oracle import dense_loglinear_attention
from .fenwick import (
    fenwick_prefix_buckets,
    hierarchical_level_index,
    max_active_levels,
    timestep_buckets,
)
from .recurrent_reference import (
    LogLinearAttentionReference,
    recurrent_loglinear_attention,
)
from .state import FenwickBucketState, LogLinearState, SingleBatchLogLinearState

__all__ = [
    "FenwickBucketState",
    "LogLinearState",
    "SingleBatchLogLinearState",
    "fenwick_prefix_buckets",
    "max_active_levels",
    "timestep_buckets",
    "hierarchical_level_index",
    "dense_loglinear_attention",
    "recurrent_loglinear_attention",
    "LogLinearAttentionReference",
]
