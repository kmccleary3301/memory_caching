from __future__ import annotations

from dataclasses import dataclass
import random
import uuid

from .scoring import exact_match


@dataclass(frozen=True)
class NIAHExample:
    task: str
    context_length: int
    prompt: str
    answer: str


def _distractor_block(rng: random.Random, length: int) -> str:
    vocab = ["orchid", "signal", "river", "matrix", "grain", "planet", "ember"]
    words = [rng.choice(vocab) for _ in range(max(length // 6, 1))]
    return " ".join(words)


def _split_by_mode(context_length: int, mode: str, rng: random.Random) -> tuple[int, int]:
    if mode == "front":
        left = context_length // 8
    elif mode == "back":
        left = context_length - (context_length // 8)
    elif mode == "middle":
        left = context_length // 2
    elif mode == "uniform":
        left = rng.randint(0, context_length)
    else:
        raise ValueError("position_mode must be one of: uniform, front, middle, back")
    right = max(context_length - left, 0)
    return left, right


def generate_niah_examples(
    *,
    task: str,
    context_length: int,
    samples: int,
    seed: int,
    position_mode: str = "uniform",
) -> list[NIAHExample]:
    if task not in {"s_niah_1", "s_niah_2", "s_niah_3"}:
        raise ValueError(f"unsupported niah task: {task}")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")

    rng = random.Random(seed)
    examples: list[NIAHExample] = []

    for _ in range(samples):
        left_len, right_len = _split_by_mode(context_length, position_mode, rng)
        left = _distractor_block(rng, left_len)
        right = _distractor_block(rng, right_len)

        if task == "s_niah_1":
            answer = f"P{rng.randint(100000, 999999)}"
            marker = f"PASSKEY: {answer}"
            question = "Return the exact PASSKEY."
        elif task == "s_niah_2":
            answer = str(rng.randint(10_000_000, 99_999_999))
            marker = f"NEEDLE_NUMBER: {answer}"
            question = "Return the exact NEEDLE_NUMBER."
        else:
            answer = str(uuid.UUID(int=rng.getrandbits(128)))
            marker = f"NEEDLE_UUID: {answer}"
            question = "Return the exact NEEDLE_UUID."

        prompt = (
            "You are given a long context.\n"
            f"{left}\n"
            f"{marker}\n"
            f"{right}\n"
            f"Question: {question}\nAnswer:"
        )
        examples.append(
            NIAHExample(
                task=task,
                context_length=context_length,
                prompt=prompt,
                answer=answer,
            )
        )
    return examples


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def score_niah(prediction: str, answer: str) -> float:
    return exact_match(prediction, answer)
