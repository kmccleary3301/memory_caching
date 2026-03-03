from __future__ import annotations

from dataclasses import dataclass


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


def score_longbench(prediction: str, answer: str) -> float:
    return 1.0 if prediction.strip().upper() == answer.strip().upper() else 0.0
