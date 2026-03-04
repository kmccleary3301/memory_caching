from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from .adapters import BenchmarkAdapter
from .longbench import LONG_BENCH_TASK_GROUPS, load_longbench_examples, score_longbench
from .mqar import generate_mqar_examples, score_mqar
from .niah import generate_niah_examples, score_niah
from .retrieval import (
    SUPPORTED_RETRIEVAL_DATASETS,
    load_retrieval_examples,
    score_retrieval,
)
from .seed import make_seed


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
    position_mode: str = "uniform",
) -> dict[str, object]:
    rows: list[TaskMetric] = []

    for adapter in adapters:
        for task in tasks:
            for length in context_lengths:
                examples = generate_niah_examples(
                    task=task,
                    context_length=length,
                    samples=samples_per_length,
                    seed=make_seed(seed, adapter.name, task, length),
                    position_mode=position_mode,
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
        "position_mode": position_mode,
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
            seed=make_seed(seed, adapter.name, num_pairs, num_queries),
        )

        micro_scores: list[float] = []
        macro_scores: list[float] = []
        for ex in examples:
            pred = adapter.predict(ex.prompt)
            micro, macro = score_mqar(pred, ex.answers)
            micro_scores.append(micro)
            macro_scores.append(macro)

        micro_acc = float(sum(micro_scores) / len(micro_scores)) if micro_scores else 0.0
        macro_acc = float(sum(macro_scores) / len(macro_scores)) if macro_scores else 0.0

        rows.append(
            {
                "adapter": adapter.name,
                "samples": len(examples),
                "num_pairs": num_pairs,
                "num_queries": num_queries,
                "micro_accuracy": micro_acc,
                "macro_accuracy": macro_acc,
            }
        )

    mean_accuracy = float(sum(r["micro_accuracy"] for r in rows) / len(rows)) if rows else 0.0
    return {
        "benchmark": "mqar",
        "mean_accuracy": mean_accuracy,
        "rows": rows,
    }


def run_longbench_suite(
    *,
    adapters: Sequence[BenchmarkAdapter],
    tasks: Sequence[str],
    samples_per_task: int,
    seed: int,
    dataset_file: str | None = None,
) -> dict[str, object]:
    for task in tasks:
        if task not in LONG_BENCH_TASK_GROUPS:
            raise ValueError(f"unknown longbench task group: {task}")

    rows: list[dict[str, object]] = []
    for adapter in adapters:
        for task in tasks:
            examples = load_longbench_examples(
                task_group=task,
                samples=samples_per_task,
                seed=make_seed(seed, adapter.name, task),
                dataset_file=dataset_file,
            )
            scores: list[float] = []
            for ex in examples:
                pred = adapter.predict(ex.prompt)
                scores.append(score_longbench(pred, ex.answer))

            rows.append(
                {
                    "adapter": adapter.name,
                    "task": task,
                    "samples": len(examples),
                    "accuracy": float(sum(scores) / len(scores)) if scores else 0.0,
                }
            )

    mean_accuracy = float(sum(r["accuracy"] for r in rows) / len(rows)) if rows else 0.0
    return {
        "benchmark": "longbench",
        "mean_accuracy": mean_accuracy,
        "rows": rows,
    }


def run_retrieval_suite(
    *,
    adapters: Sequence[BenchmarkAdapter],
    datasets: Sequence[str],
    truncation_lengths: Sequence[int],
    samples_per_dataset: int,
    seed: int,
    dataset_file: str | None = None,
) -> dict[str, object]:
    if any(length <= 0 for length in truncation_lengths):
        raise ValueError("all truncation lengths must be positive")

    for ds in datasets:
        if ds not in SUPPORTED_RETRIEVAL_DATASETS:
            raise ValueError(f"unsupported retrieval dataset: {ds}")

    rows: list[dict[str, object]] = []
    for adapter in adapters:
        for ds in datasets:
            for tlen in truncation_lengths:
                examples = load_retrieval_examples(
                    dataset=ds,
                    truncation_length=tlen,
                    samples=samples_per_dataset,
                    seed=make_seed(seed, adapter.name, ds, tlen),
                    dataset_file=dataset_file,
                )
                scores: list[float] = []
                for ex in examples:
                    pred = adapter.predict(ex.prompt)
                    scores.append(score_retrieval(pred, ex.answer))

                rows.append(
                    {
                        "adapter": adapter.name,
                        "dataset": ds,
                        "truncation_length": tlen,
                        "samples": len(examples),
                        "accuracy": float(sum(scores) / len(scores)) if scores else 0.0,
                    }
                )

    mean_accuracy = float(sum(r["accuracy"] for r in rows) / len(rows)) if rows else 0.0
    return {
        "benchmark": "retrieval",
        "mean_accuracy": mean_accuracy,
        "rows": rows,
    }


RunnerFn = Callable[..., dict[str, Any]]


_RUNNERS: dict[str, RunnerFn] = {
    "niah": run_niah_suite,
    "mqar": run_mqar_suite,
    "longbench": run_longbench_suite,
    "retrieval": run_retrieval_suite,
}


def list_runners() -> list[str]:
    return sorted(_RUNNERS.keys())


def get_runner(name: str) -> RunnerFn:
    key = name.strip().lower()
    if key not in _RUNNERS:
        raise ValueError(f"unknown runner: {name}")
    return _RUNNERS[key]
