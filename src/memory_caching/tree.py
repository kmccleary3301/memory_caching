from __future__ import annotations

from typing import Any

import torch


Tree = Any


def tree_clone(obj: Tree) -> Tree:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
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
    if isinstance(obj, list):
        return [tree_detach_clone(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(tree_detach_clone(x) for x in obj)
    if isinstance(obj, dict):
        return {k: tree_detach_clone(v) for k, v in obj.items()}
    return obj
