"""Memory Caching reproduction package."""

from .backends.dla import DLABackend
from .backends.linear import LinearMemoryBackend
from .backends.swla import SWLABackend
from .backends.titans import TitansBackend
from .config import DLAConfig, MCConfig, SWLAConfig, TitansConfig
from .layer import MemoryCachingLayer
from .smoke import run_smoke_eval, run_smoke_train

__all__ = [
    "__version__",
    "DLAConfig",
    "TitansConfig",
    "SWLAConfig",
    "MCConfig",
    "MemoryCachingLayer",
    "LinearMemoryBackend",
    "DLABackend",
    "SWLABackend",
    "TitansBackend",
    "run_smoke_train",
    "run_smoke_eval",
]

__version__ = "0.1.0"
