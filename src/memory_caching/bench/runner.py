from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .adapters import BenchmarkAdapter
from .mqar import generate_mqar_examples, score_mqar
from .niah import generate_niah_examples, score_niah


@dataclass(frozen=True)
class TaskMetric:
    adapter: str
    task: str
    context_length: int
    samples: int
    accuracy: float


def run_niah_suite(
    *,
    adapters: Sequence[BenchmarkAdapter],
    tasks: Sequence[str],
    context_lengths: Sequence[int],
    samples_per_length: int,
    seed: int,
) -> dict[str, object]:
    rows: list[TaskMetric] = []

    for adapter in adapters:
        for task in tasks:
            for length in context_lengths:
                examples = generate_niah_examples(
                    task=task,
                    context_length=length,
                    samples=samples_per_length,
                    seed=seed + length,
                )
                scores = [
                    score_niah(adapter.predict(ex.prompt), ex.answer) for ex in examples
                ]
                acc = float(sum(scores) / len(scores)) if scores else 0.0
                rows.append(
                    TaskMetric(
                        adapter=adapter.name,
                        task=task,
                        context_length=length,
                        samples=len(examples),
                        accuracy=acc,
                    )
                )

    mean_accuracy = float(sum(r.accuracy for r in rows) / len(rows)) if rows else 0.0
    return {
        "benchmark": "niah",
        "mean_accuracy": mean_accuracy,
        "rows": [asdict(r) for r in rows],
    }


def run_mqar_suite(
    *,
    adapters: Sequence[BenchmarkAdapter],
    samples: int,
    num_pairs: int,
    num_queries: int,
    seed: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []

    for adapter in adapters:
        examples = generate_mqar_examples(
            samples=samples,
            num_pairs=num_pairs,
            num_queries=num_queries,
            seed=seed,
        )
        scores = [
            score_mqar(adapter.predict(ex.prompt), ex.answer)
            for ex in examples
        ]
        acc = float(sum(scores) / len(scores)) if scores else 0.0
        rows.append(
            {
                "adapter": adapter.name,
                "samples": len(examples),
                "num_pairs": num_pairs,
                "num_queries": num_queries,
                "accuracy": acc,
            }
        )

    mean_accuracy = float(sum(r["accuracy"] for r in rows) / len(rows)) if rows else 0.0
    return {
        "benchmark": "mqar",
        "mean_accuracy": mean_accuracy,
        "rows": rows,
    }
