from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import torch


Tree = Any


def _clone_dataclass(obj: Any, *, detach: bool) -> Any:
    values: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        values[f.name] = tree_detach_clone(value) if detach else tree_clone(value)
    return obj.__class__(**values)


def tree_clone(obj: Tree) -> Tree:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if is_dataclass(obj):
        return _clone_dataclass(obj, detach=False)
    if isinstance(obj, list):
        return [tree_clone(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(tree_clone(x) for x in obj)
    if isinstance(obj, dict):
        return {k: tree_clone(v) for k, v in obj.items()}
    return obj


def tree_detach_clone(obj: Tree) -> Tree:
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if is_dataclass(obj):
        return _clone_dataclass(obj, detach=True)
    if isinstance(obj, list):
        return [tree_detach_clone(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(tree_detach_clone(x) for x in obj)
    if isinstance(obj, dict):
        return {k: tree_detach_clone(v) for k, v in obj.items()}
    return obj
