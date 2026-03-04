from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any


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


def _truncate_text(text: str, truncation_length: int) -> str:
    if len(text) <= truncation_length:
        return text
    return text[:truncation_length]


def _extract_retrieval_answer(row: dict[str, Any]) -> str:
    if "answer" in row and isinstance(row["answer"], str):
        return row["answer"]
    answers = row.get("answers")
    if isinstance(answers, list) and len(answers) > 0 and isinstance(answers[0], str):
        return answers[0]
    raise ValueError("row missing answer/answers")


def load_retrieval_examples(
    *,
    dataset: str,
    truncation_length: int,
    samples: int,
    seed: int,
    dataset_file: str | None,
) -> list[RetrievalExample]:
    if dataset not in SUPPORTED_RETRIEVAL_DATASETS:
        raise ValueError(f"unsupported retrieval dataset: {dataset}")
    if truncation_length <= 0:
        raise ValueError("truncation_length must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")

    if dataset_file is None:
        return [
            RetrievalExample(
                dataset=dataset,
                truncation_length=truncation_length,
                prompt=build_retrieval_prompt(dataset, truncation_length, i),
                answer="RETRIEVAL_OK",
            )
            for i in range(samples)
        ]

    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        raise ValueError(f"dataset_file does not exist: {dataset_file}")

    rows: list[RetrievalExample] = []
    for line in dataset_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        row_dataset = str(payload.get("dataset", "")).strip().lower()
        if row_dataset != dataset:
            continue

        document = str(payload.get("document", payload.get("context", "")))
        question = str(payload.get("question", ""))
        if not question.strip():
            continue
        answer = _extract_retrieval_answer(payload)
        truncated_doc = _truncate_text(document, truncation_length)
        prompt = (
            f"DATASET: {dataset}\n"
            f"TRUNCATION: {truncation_length}\n"
            f"DOC: {truncated_doc}\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )
        rows.append(
            RetrievalExample(
                dataset=dataset,
                truncation_length=truncation_length,
                prompt=prompt,
                answer=answer,
            )
        )

    if len(rows) < samples:
        raise ValueError(
            f"dataset_file={dataset_file} has only {len(rows)} rows for dataset={dataset}, need {samples}"
        )

    picker = random.Random(seed)
    if len(rows) == samples:
        return rows
    return picker.sample(rows, k=samples)


def score_retrieval(prediction: str, answer: str) -> float:
    return 1.0 if prediction.strip().upper() == answer.strip().upper() else 0.0
