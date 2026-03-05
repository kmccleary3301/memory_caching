from __future__ import annotations

import json

import typer

from .bench.adapters import (
    BenchmarkAdapter,
    DLAMCAdapter,
    LinearMCAdapter,
    ModelBackedAdapter,
    TitansMCAdapter,
)
from .bench.artifacts import create_bundle, write_artifacts
from .bench.runner import get_runner, list_runners, run_mqar_suite, run_niah_suite
from .segmentation import constant_segments, logarithmic_segments, spans_from_lengths
from .smoke import run_smoke_eval, run_smoke_train

app = typer.Typer(help="Memory Caching reproduction CLI")
bench_app = typer.Typer(help="Benchmark harness commands")
app.add_typer(bench_app, name="bench")


def _select_adapters(which: str):
    normalized = which.strip().lower()
    if normalized == "linear":
        return [LinearMCAdapter()]
    if normalized == "dla":
        return [DLAMCAdapter()]
    if normalized == "titans":
        return [TitansMCAdapter()]
    if normalized == "both":
        return [LinearMCAdapter(), DLAMCAdapter()]
    if normalized == "all":
        return [LinearMCAdapter(), DLAMCAdapter(), TitansMCAdapter()]
    raise typer.BadParameter("adapter must be one of: linear, dla, titans, both, all")


def _adapter_type(adapters: list[BenchmarkAdapter]) -> str:
    if adapters and all(isinstance(adapter, ModelBackedAdapter) for adapter in adapters):
        return "model_backed"
    return "rule_based"


def _warn_if_rule_based(adapter_type: str) -> None:
    if adapter_type == "rule_based":
        typer.secho(
            "WARNING: benchmark adapters are rule-based compatibility adapters, not model-backed evaluators.",
            fg=typer.colors.YELLOW,
            err=True,
        )


@app.command()
def status() -> None:
    typer.echo("memory_caching: m70 execution in progress")


@app.command("list-variants")
def list_variants() -> None:
    typer.echo("backend: linear, dla, titans, swla")
    typer.echo("aggregation: residual, grm, soup, ssc")
    typer.echo("segmentation: constant, logarithmic")
    typer.echo("state_init_mode: checkpoint, restart")


@app.command()
def segment(
    length: int = typer.Option(..., help="Sequence length"),
    mode: str = typer.Option("constant", help="constant or logarithmic"),
    segment_size: int = typer.Option(256, help="Constant segment size"),
) -> None:
    if length <= 0:
        raise typer.BadParameter("length must be positive")

    mode_norm = mode.lower().strip()
    if mode_norm == "constant":
        spans = constant_segments(length, segment_size)
    elif mode_norm in {"log", "logarithmic"}:
        lengths = logarithmic_segments(length)
        spans = spans_from_lengths(lengths)
    else:
        raise typer.BadParameter("mode must be one of: constant, logarithmic")

    for idx, (start, end) in enumerate(spans, start=1):
        typer.echo(f"segment_{idx}: [{start}, {end}) len={end - start}")


@app.command("smoke-train")
def smoke_train(
    steps: int = typer.Option(20),
    batch_size: int = typer.Option(4),
    seq_len: int = typer.Option(64),
    vocab_size: int = typer.Option(128),
    d_model: int = typer.Option(64),
    num_heads: int = typer.Option(4),
    backend: str = typer.Option("linear"),
    dla_memory_width: int = typer.Option(64),
    dla_memory_depth: int = typer.Option(2),
    dla_objective: str = typer.Option("dot"),
    dla_inner_update_mode: str = typer.Option("stopgrad"),
    dla_step_size: float = typer.Option(0.05),
    dla_momentum: float = typer.Option(0.0),
    titans_memory_width: int = typer.Option(64),
    titans_memory_depth: int = typer.Option(2),
    titans_objective: str = typer.Option("l2"),
    titans_inner_update_mode: str = typer.Option("stopgrad"),
    titans_step_size: float = typer.Option(0.05),
    titans_momentum: float = typer.Option(0.9),
    titans_retention_alpha: float = typer.Option(1.0),
    titans_update_convention: str = typer.Option("paper"),
    swla_alpha: float = typer.Option(1.0),
    swla_beta: float = typer.Option(0.0),
    swla_lam: float = typer.Option(1.0),
    segment_size: int = typer.Option(16),
    aggregation: str = typer.Option("grm"),
    segmentation: str = typer.Option("constant"),
    state_init_mode: str = typer.Option("checkpoint"),
    ssc_top_k: int = typer.Option(2),
    lr: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto"),
    out_json: str | None = typer.Option(None),
) -> None:
    metrics = run_smoke_train(
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        backend=backend,
        dla_memory_width=dla_memory_width,
        dla_memory_depth=dla_memory_depth,
        dla_objective=dla_objective,
        dla_inner_update_mode=dla_inner_update_mode,
        dla_step_size=dla_step_size,
        dla_momentum=dla_momentum,
        titans_memory_width=titans_memory_width,
        titans_memory_depth=titans_memory_depth,
        titans_objective=titans_objective,
        titans_inner_update_mode=titans_inner_update_mode,
        titans_step_size=titans_step_size,
        titans_momentum=titans_momentum,
        titans_retention_alpha=titans_retention_alpha,
        titans_update_convention=titans_update_convention,
        swla_alpha=swla_alpha,
        swla_beta=swla_beta,
        swla_lam=swla_lam,
        segment_size=segment_size,
        aggregation=aggregation,
        segmentation=segmentation,
        state_init_mode=state_init_mode,
        ssc_top_k=ssc_top_k,
        lr=lr,
        seed=seed,
        device=device,
        out_json=out_json,
    )
    typer.echo(json.dumps(metrics, sort_keys=True, indent=2))


@app.command("smoke-eval")
def smoke_eval(
    warmup_steps: int = typer.Option(0),
    batch_size: int = typer.Option(4),
    seq_len: int = typer.Option(64),
    vocab_size: int = typer.Option(128),
    d_model: int = typer.Option(64),
    num_heads: int = typer.Option(4),
    backend: str = typer.Option("linear"),
    dla_memory_width: int = typer.Option(64),
    dla_memory_depth: int = typer.Option(2),
    dla_objective: str = typer.Option("dot"),
    dla_inner_update_mode: str = typer.Option("stopgrad"),
    dla_step_size: float = typer.Option(0.05),
    dla_momentum: float = typer.Option(0.0),
    titans_memory_width: int = typer.Option(64),
    titans_memory_depth: int = typer.Option(2),
    titans_objective: str = typer.Option("l2"),
    titans_inner_update_mode: str = typer.Option("stopgrad"),
    titans_step_size: float = typer.Option(0.05),
    titans_momentum: float = typer.Option(0.9),
    titans_retention_alpha: float = typer.Option(1.0),
    titans_update_convention: str = typer.Option("paper"),
    swla_alpha: float = typer.Option(1.0),
    swla_beta: float = typer.Option(0.0),
    swla_lam: float = typer.Option(1.0),
    segment_size: int = typer.Option(16),
    aggregation: str = typer.Option("grm"),
    segmentation: str = typer.Option("constant"),
    state_init_mode: str = typer.Option("checkpoint"),
    ssc_top_k: int = typer.Option(2),
    lr: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto"),
    out_json: str | None = typer.Option(None),
) -> None:
    metrics = run_smoke_eval(
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        backend=backend,
        dla_memory_width=dla_memory_width,
        dla_memory_depth=dla_memory_depth,
        dla_objective=dla_objective,
        dla_inner_update_mode=dla_inner_update_mode,
        dla_step_size=dla_step_size,
        dla_momentum=dla_momentum,
        titans_memory_width=titans_memory_width,
        titans_memory_depth=titans_memory_depth,
        titans_objective=titans_objective,
        titans_inner_update_mode=titans_inner_update_mode,
        titans_step_size=titans_step_size,
        titans_momentum=titans_momentum,
        titans_retention_alpha=titans_retention_alpha,
        titans_update_convention=titans_update_convention,
        swla_alpha=swla_alpha,
        swla_beta=swla_beta,
        swla_lam=swla_lam,
        segment_size=segment_size,
        aggregation=aggregation,
        segmentation=segmentation,
        state_init_mode=state_init_mode,
        ssc_top_k=ssc_top_k,
        lr=lr,
        seed=seed,
        device=device,
        out_json=out_json,
    )
    typer.echo(json.dumps(metrics, sort_keys=True, indent=2))


@bench_app.command("list")
def bench_list() -> None:
    typer.echo("runners: " + ", ".join(list_runners()))


@bench_app.command("niah")
def bench_niah(
    adapter: str = typer.Option("all", help="linear, dla, titans, both, or all"),
    tasks: str = typer.Option("s_niah_1,s_niah_2,s_niah_3"),
    context_lengths: str = typer.Option("4096,8192,16384"),
    samples_per_length: int = typer.Option(16),
    seed: int = typer.Option(0),
    position_mode: str = typer.Option("uniform"),
    out_dir: str | None = typer.Option(None),
) -> None:
    adapters = _select_adapters(adapter)
    adapter_type = _adapter_type(adapters)
    _warn_if_rule_based(adapter_type)
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    lengths = [int(x.strip()) for x in context_lengths.split(",") if x.strip()]

    result = run_niah_suite(
        adapters=adapters,
        tasks=task_list,
        context_lengths=lengths,
        samples_per_length=samples_per_length,
        seed=seed,
        position_mode=position_mode,
    )
    result = {**result, "adapter_type": adapter_type}

    bundle = create_bundle(out_dir)
    write_artifacts(
        bundle=bundle,
        run_type="niah",
        config={
            "adapter": adapter,
            "tasks": task_list,
            "context_lengths": lengths,
            "samples_per_length": samples_per_length,
            "seed": seed,
            "position_mode": position_mode,
            "adapter_type": adapter_type,
        },
        metrics=result,
        runner_version="v0.2",
        dataset_revision="synthetic-v2",
    )
    typer.echo(json.dumps({**result, "artifact_dir": str(bundle.root_dir)}, indent=2))


@bench_app.command("mqar")
def bench_mqar(
    adapter: str = typer.Option("all", help="linear, dla, titans, both, or all"),
    samples: int = typer.Option(64),
    num_pairs: int = typer.Option(16),
    num_queries: int = typer.Option(4),
    pair_grid: str | None = typer.Option(None, help="comma list, e.g. 8,16,32"),
    query_grid: str | None = typer.Option(None, help="comma list, e.g. 1,4,8"),
    seed: int = typer.Option(0),
    out_dir: str | None = typer.Option(None),
) -> None:
    adapters = _select_adapters(adapter)
    adapter_type = _adapter_type(adapters)
    _warn_if_rule_based(adapter_type)

    pair_values = [num_pairs]
    query_values = [num_queries]
    if pair_grid:
        pair_values = [int(x.strip()) for x in pair_grid.split(",") if x.strip()]
    if query_grid:
        query_values = [int(x.strip()) for x in query_grid.split(",") if x.strip()]

    rows = []
    for p in pair_values:
        for q in query_values:
            result = run_mqar_suite(
                adapters=adapters,
                samples=samples,
                num_pairs=p,
                num_queries=q,
                seed=seed,
            )
            for row in result["rows"]:
                rows.append(row)

    result = {
        "benchmark": "mqar",
        "mean_accuracy": float(sum(r["micro_accuracy"] for r in rows) / len(rows)) if rows else 0.0,
        "rows": rows,
        "adapter_type": adapter_type,
    }

    bundle = create_bundle(out_dir)
    write_artifacts(
        bundle=bundle,
        run_type="mqar",
        config={
            "adapter": adapter,
            "samples": samples,
            "pair_grid": pair_values,
            "query_grid": query_values,
            "seed": seed,
            "adapter_type": adapter_type,
        },
        metrics=result,
        runner_version="v0.2",
        dataset_revision="synthetic-v2",
    )
    typer.echo(json.dumps({**result, "artifact_dir": str(bundle.root_dir)}, indent=2))


@bench_app.command("longbench")
def bench_longbench(
    adapter: str = typer.Option("all"),
    tasks: str = typer.Option("single_doc_qa,multi_doc_qa,summarization,few_shot,code"),
    samples_per_task: int = typer.Option(4),
    seed: int = typer.Option(0),
    dataset_file: str | None = typer.Option(
        None,
        help="optional JSONL dataset file with task_group/prompt/answer fields",
    ),
    out_dir: str | None = typer.Option(None),
) -> None:
    adapters = _select_adapters(adapter)
    adapter_type = _adapter_type(adapters)
    _warn_if_rule_based(adapter_type)
    runner = get_runner("longbench")
    result = runner(
        adapters=adapters,
        tasks=[t.strip() for t in tasks.split(",") if t.strip()],
        samples_per_task=samples_per_task,
        seed=seed,
        dataset_file=dataset_file,
    )
    result = {**result, "adapter_type": adapter_type}

    bundle = create_bundle(out_dir)
    write_artifacts(
        bundle=bundle,
        run_type="longbench",
        config={
            "adapter": adapter,
            "tasks": tasks,
            "samples_per_task": samples_per_task,
            "seed": seed,
            "dataset_file": dataset_file,
            "adapter_type": adapter_type,
        },
        metrics=result,
        runner_version="v0.2",
        dataset_revision="dataset-file-v1" if dataset_file else "scaffold-v1",
    )
    typer.echo(json.dumps({**result, "artifact_dir": str(bundle.root_dir)}, indent=2))


@bench_app.command("retrieval")
def bench_retrieval(
    adapter: str = typer.Option("all"),
    datasets: str = typer.Option("swde,squad,fda"),
    truncation_lengths: str = typer.Option("512,1024,2048,16384"),
    samples_per_dataset: int = typer.Option(4),
    seed: int = typer.Option(0),
    dataset_file: str | None = typer.Option(
        None,
        help="optional JSONL dataset file with dataset/document/question/answer fields",
    ),
    out_dir: str | None = typer.Option(None),
) -> None:
    adapters = _select_adapters(adapter)
    adapter_type = _adapter_type(adapters)
    _warn_if_rule_based(adapter_type)
    runner = get_runner("retrieval")
    result = runner(
        adapters=adapters,
        datasets=[d.strip() for d in datasets.split(",") if d.strip()],
        truncation_lengths=[int(x.strip()) for x in truncation_lengths.split(",") if x.strip()],
        samples_per_dataset=samples_per_dataset,
        seed=seed,
        dataset_file=dataset_file,
    )
    result = {**result, "adapter_type": adapter_type}

    bundle = create_bundle(out_dir)
    write_artifacts(
        bundle=bundle,
        run_type="retrieval",
        config={
            "adapter": adapter,
            "datasets": datasets,
            "truncation_lengths": truncation_lengths,
            "samples_per_dataset": samples_per_dataset,
            "seed": seed,
            "dataset_file": dataset_file,
            "adapter_type": adapter_type,
        },
        metrics=result,
        runner_version="v0.2",
        dataset_revision="dataset-file-v1" if dataset_file else "scaffold-v1",
    )
    typer.echo(json.dumps({**result, "artifact_dir": str(bundle.root_dir)}, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
