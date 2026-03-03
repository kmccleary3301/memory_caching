from __future__ import annotations

import json

import typer

from .bench.adapters import DLAMCAdapter, LinearMCAdapter
from .bench.artifacts import create_bundle, write_artifacts
from .bench.runner import run_mqar_suite, run_niah_suite
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
    if normalized == "both":
        return [LinearMCAdapter(), DLAMCAdapter()]
    raise typer.BadParameter("adapter must be one of: linear, dla, both")


@app.command()
def status() -> None:
    """Print repository status."""
    typer.echo("memory_caching: phase-1 mc core + dla + benchmark harness ready")


@app.command("list-variants")
def list_variants() -> None:
    """Print available variants."""
    typer.echo("backend: linear, dla")
    typer.echo("aggregation: residual, grm, soup, ssc")
    typer.echo("segmentation: constant, logarithmic")
    typer.echo("state_init_mode: checkpoint, restart")


@app.command()
def segment(
    length: int = typer.Option(..., help="Sequence length"),
    mode: str = typer.Option("constant", help="constant or logarithmic"),
    segment_size: int = typer.Option(256, help="Constant segment size"),
) -> None:
    """Print segment boundaries for a sequence length."""
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
    """Run minimal synthetic MC training smoke loop and print JSON metrics."""
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
    """Run minimal synthetic MC eval smoke loop and print JSON metrics."""
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


@bench_app.command("niah")
def bench_niah(
    adapter: str = typer.Option("both", help="linear, dla, or both"),
    tasks: str = typer.Option("s_niah_1,s_niah_2,s_niah_3"),
    context_lengths: str = typer.Option("4096,8192,16384"),
    samples_per_length: int = typer.Option(16),
    seed: int = typer.Option(0),
    out_dir: str | None = typer.Option(None),
) -> None:
    adapters = _select_adapters(adapter)
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    lengths = [int(x.strip()) for x in context_lengths.split(",") if x.strip()]

    result = run_niah_suite(
        adapters=adapters,
        tasks=task_list,
        context_lengths=lengths,
        samples_per_length=samples_per_length,
        seed=seed,
    )

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
        },
        metrics=result,
    )
    typer.echo(json.dumps({**result, "artifact_dir": str(bundle.root_dir)}, indent=2))


@bench_app.command("mqar")
def bench_mqar(
    adapter: str = typer.Option("both", help="linear, dla, or both"),
    samples: int = typer.Option(64),
    num_pairs: int = typer.Option(16),
    num_queries: int = typer.Option(4),
    seed: int = typer.Option(0),
    out_dir: str | None = typer.Option(None),
) -> None:
    adapters = _select_adapters(adapter)
    result = run_mqar_suite(
        adapters=adapters,
        samples=samples,
        num_pairs=num_pairs,
        num_queries=num_queries,
        seed=seed,
    )

    bundle = create_bundle(out_dir)
    write_artifacts(
        bundle=bundle,
        run_type="mqar",
        config={
            "adapter": adapter,
            "samples": samples,
            "num_pairs": num_pairs,
            "num_queries": num_queries,
            "seed": seed,
        },
        metrics=result,
    )
    typer.echo(json.dumps({**result, "artifact_dir": str(bundle.root_dir)}, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
