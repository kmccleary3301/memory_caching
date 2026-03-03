from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_RETRIEVAL_DATASETS = {"swde", "squad", "fda"}


@dataclass(frozen=True)
class RetrievalExample:
    dataset: str
    truncation_length: int
    prompt: str
    answer: str


def build_retrieval_prompt(dataset: str, truncation_length: int, sample_idx: int) -> str:
    if dataset not in SUPPORTED_RETRIEVAL_DATASETS:
        raise ValueError(f"unsupported retrieval dataset: {dataset}")
    if truncation_length <= 0:
        raise ValueError("truncation_length must be positive")
    return (
        f"DATASET: {dataset}\n"
        f"TRUNCATION: {truncation_length}\n"
        f"DOC: synthetic retrieval sample {sample_idx}\n"
        "QUESTION: return token RETRIEVAL_OK\n"
        "ANSWER:"
    )


def score_retrieval(prediction: str, answer: str) -> float:
    return 1.0 if prediction.strip().upper() == answer.strip().upper() else 0.0
