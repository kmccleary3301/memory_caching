from __future__ import annotations

from dataclasses import dataclass
import random
import uuid


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


def generate_niah_examples(
    *,
    task: str,
    context_length: int,
    samples: int,
    seed: int,
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
        left = _distractor_block(rng, context_length // 2)
        right = _distractor_block(rng, context_length // 2)

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


def score_niah(prediction: str, answer: str) -> float:
    return 1.0 if prediction.strip().lower() == answer.strip().lower() else 0.0
