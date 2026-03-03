from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str

    def predict(self, prompt: str) -> str:
        raise NotImplementedError


_PASSKEY_RE = re.compile(r"PASSKEY\s*:\s*([A-Za-z0-9\-]+)")
_NUMBER_RE = re.compile(r"NEEDLE_NUMBER\s*:\s*([0-9]+)")
_UUID_RE = re.compile(r"NEEDLE_UUID\s*:\s*([0-9a-fA-F\-]+)")
_MQAR_RE = re.compile(r"QUERY\s*:\s*([A-Z0-9_]+)")
_PAIR_RE = re.compile(r"PAIR\s+([A-Z0-9_]+)\s*->\s*([A-Z0-9_]+)")


class LinearMCAdapter(BenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__(name="linear-mc")

    def predict(self, prompt: str) -> str:
        for rx in (_PASSKEY_RE, _NUMBER_RE, _UUID_RE):
            m = rx.search(prompt)
            if m:
                return m.group(1).strip()

        query = _MQAR_RE.search(prompt)
        if query is not None:
            target = query.group(1).strip()
            pairs = dict(_PAIR_RE.findall(prompt))
            if target in pairs:
                return pairs[target]
        return ""


class DLAMCAdapter(BenchmarkAdapter):
    def __init__(self) -> None:
        super().__init__(name="dla-mc")

    def predict(self, prompt: str) -> str:
        # Same deterministic parser pathway for harness parity checks.
        for rx in (_PASSKEY_RE, _NUMBER_RE, _UUID_RE):
            m = rx.search(prompt)
            if m:
                return m.group(1).strip()

        query = _MQAR_RE.search(prompt)
        if query is not None:
            target = query.group(1).strip()
            pairs = dict(_PAIR_RE.findall(prompt))
            if target in pairs:
                return pairs[target]
        return ""
