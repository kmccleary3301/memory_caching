from __future__ import annotations

import os
import random
import subprocess
import sys

from memory_caching.bench.seed import make_seed


def test_make_seed_is_pure() -> None:
    before = random.getstate()
    _ = make_seed(0, "linear-mc", "s_niah_1", 4096)
    after = random.getstate()
    assert before == after


def test_make_seed_is_stable_within_process() -> None:
    first = make_seed(0, "linear-mc", "s_niah_1", 4096)
    second = make_seed(0, "linear-mc", "s_niah_1", 4096)
    assert first == second


def _subprocess_seed(pyhashseed: str) -> int:
    code = (
        "from memory_caching.bench.seed import make_seed; "
        "print(make_seed(0, 'linear-mc', 's_niah_1', 4096))"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = pyhashseed
    out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    return int(out.strip())


def test_make_seed_is_process_stable() -> None:
    first = _subprocess_seed("1")
    second = _subprocess_seed("999")
    assert first == second
