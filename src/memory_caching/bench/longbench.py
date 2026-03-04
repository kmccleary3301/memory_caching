from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any


LONG_BENCH_TASK_GROUPS = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa"],
    "multi_doc_qa": ["hotpotqa", "2wikimultihopqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shot": ["trec", "triviaqa", "samsum"],
    "code": ["lcc", "repobench_p"],
}


@dataclass(frozen=True)
class LongBenchExample:
    task_group: str
    prompt: str
    answer: str


def build_longbench_prompt(task_group: str, sample_idx: int) -> str:
    if task_group not in LONG_BENCH_TASK_GROUPS:
        raise ValueError(f"unknown longbench task group: {task_group}")
    return (
        f"TASK_GROUP: {task_group}\n"
        f"DOC: synthetic longbench sample {sample_idx}\n"
        "QUESTION: return token ANSWER_OK\n"
        "ANSWER:"
    )


def _extract_answer(row: dict[str, Any]) -> str:
    if "answer" in row and isinstance(row["answer"], str):
        return row["answer"]
    answers = row.get("answers")
    if isinstance(answers, list) and len(answers) > 0 and isinstance(answers[0], str):
        return answers[0]
    raise ValueError("row missing answer/answers")


def load_longbench_examples(
    *,
    task_group: str,
    samples: int,
    seed: int,
    dataset_file: str | None,
) -> list[LongBenchExample]:
    if task_group not in LONG_BENCH_TASK_GROUPS:
        raise ValueError(f"unknown longbench task group: {task_group}")
    if samples <= 0:
        raise ValueError("samples must be positive")

    if dataset_file is None:
        return [
            LongBenchExample(
                task_group=task_group,
                prompt=build_longbench_prompt(task_group, i),
                answer="ANSWER_OK",
            )
            for i in range(samples)
        ]

    rows: list[LongBenchExample] = []
    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        raise ValueError(f"dataset_file does not exist: {dataset_file}")

    for line in dataset_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        row_group = str(payload.get("task_group", payload.get("task", ""))).strip()
        if row_group != task_group:
            continue
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        answer = _extract_answer(payload)
        rows.append(
            LongBenchExample(
                task_group=task_group,
                prompt=prompt,
                answer=answer,
            )
        )

    if len(rows) < samples:
        raise ValueError(
            f"dataset_file={dataset_file} has only {len(rows)} rows for task_group={task_group}, need {samples}"
        )

    picker = random.Random(seed)
    if len(rows) == samples:
        return rows
    return picker.sample(rows, k=samples)


def score_longbench(prediction: str, answer: str) -> float:
    return 1.0 if prediction.strip().upper() == answer.strip().upper() else 0.0
