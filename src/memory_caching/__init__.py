"""Memory Caching reproduction package."""

from .backends.linear import LinearMemoryBackend
from .config import MCConfig
from .layer import MemoryCachingLayer
from .smoke import run_smoke_eval, run_smoke_train

__all__ = [
    "__version__",
    "MCConfig",
    "MemoryCachingLayer",
    "LinearMemoryBackend",
    "run_smoke_train",
    "run_smoke_eval",
]

__version__ = "0.1.0"
