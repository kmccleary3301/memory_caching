"""Stable public Memory Caching package surface.

Top-level exports are limited to the supported runtime API. CLI entrypoints,
smoke helpers, benchmark runners, and report-generation tooling remain repo
tooling and are intentionally excluded from this surface.
"""

from .backends.dla import DLABackend
from .backends.linear import LinearMemoryBackend
from .backends.swla import SWLABackend
from .backends.titans import TitansBackend
from .config import DLAConfig, MCConfig, SWLAConfig, TitansConfig
from .layer import MemoryCachingLayer, SegmentCache

__all__ = [
    "__version__",
    "DLAConfig",
    "TitansConfig",
    "SWLAConfig",
    "MCConfig",
    "MemoryCachingLayer",
    "SegmentCache",
    "LinearMemoryBackend",
    "DLABackend",
    "SWLABackend",
    "TitansBackend",
]

__version__ = "0.1.0"
