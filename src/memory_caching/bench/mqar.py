from __future__ import annotations

from dataclasses import dataclass
import random

from .scoring import exact_match, extract_answer_candidates


@dataclass(frozen=True)
class MQARExample:
    prompt: str
    answers: tuple[str, ...]


def _token(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"


def generate_mqar_examples(
    *,
    samples: int,
    num_pairs: int,
    num_queries: int,
    seed: int,
) -> list[MQARExample]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")

    rng = random.Random(seed)
    out: list[MQARExample] = []

    for _ in range(samples):
        keys = [_token("K", i) for i in range(num_pairs)]
        values = [_token("V", i) for i in range(num_pairs)]
        pairs = list(zip(keys, values))
        rng.shuffle(pairs)

        chosen = rng.sample(pairs, k=min(num_queries, len(pairs)))
        query_lines = "\n".join([f"QUERY: {k}" for k, _ in chosen])
        answers = tuple(v for _, v in chosen)

        pair_lines = "\n".join([f"PAIR {k} -> {v}" for k, v in pairs])
        prompt = (
            "Memorize all PAIR lines, then answer each QUERY value exactly.\n"
            f"{pair_lines}\n"
            f"{query_lines}\n"
            "ANSWER:"
        )
        out.append(MQARExample(prompt=prompt, answers=answers))
    return out


def score_mqar(prediction: str, answers: tuple[str, ...]) -> tuple[float, float]:
    if len(answers) == 0:
        return 0.0, 0.0

    candidates = extract_answer_candidates(prediction)
    if len(candidates) == 0:
        return 0.0, 0.0

    available = [candidate for candidate in candidates]
    flags: list[float] = []
    for answer in answers:
        match_index = -1
        for idx, candidate in enumerate(available):
            if exact_match(candidate, answer) == 1.0:
                match_index = idx
                break
        if match_index < 0:
            flags.append(0.0)
            continue
        flags.append(1.0)
        available.pop(match_index)

    micro = float(sum(flags) / len(flags)) if flags else 0.0
    macro = 1.0 if flags and all(flag == 1.0 for flag in flags) else 0.0
    return micro, macro
