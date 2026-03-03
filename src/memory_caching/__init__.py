"""Memory Caching reproduction package."""

from .backends.dla import DLABackend
from .backends.linear import LinearMemoryBackend
from .config import DLAConfig, MCConfig
from .layer import MemoryCachingLayer
from .smoke import run_smoke_eval, run_smoke_train

__all__ = [
    "__version__",
    "DLAConfig",
    "MCConfig",
    "MemoryCachingLayer",
    "LinearMemoryBackend",
    "DLABackend",
    "run_smoke_train",
    "run_smoke_eval",
]

__version__ = "0.1.0"
