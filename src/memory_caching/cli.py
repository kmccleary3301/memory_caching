from __future__ import annotations

import json

import typer

from .segmentation import constant_segments, logarithmic_segments, spans_from_lengths
from .smoke import run_smoke_eval, run_smoke_train

app = typer.Typer(help="Memory Caching reproduction CLI")


@app.command()
def status() -> None:
    """Print repository status."""
    typer.echo("memory_caching: phase-1 MC core + smoke harness ready")


@app.command("list-variants")
def list_variants() -> None:
    """Print available aggregation and segmentation variants."""
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
    segment_size: int = typer.Option(16),
    aggregation: str = typer.Option("grm"),
    segmentation: str = typer.Option("constant"),
    state_init_mode: str = typer.Option("checkpoint"),
    ssc_top_k: int = typer.Option(2),
    lr: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto"),
) -> None:
    """Run minimal synthetic MC training smoke loop and print JSON metrics."""
    metrics = run_smoke_train(
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        segment_size=segment_size,
        aggregation=aggregation,
        segmentation=segmentation,
        state_init_mode=state_init_mode,
        ssc_top_k=ssc_top_k,
        lr=lr,
        seed=seed,
        device=device,
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
    segment_size: int = typer.Option(16),
    aggregation: str = typer.Option("grm"),
    segmentation: str = typer.Option("constant"),
    state_init_mode: str = typer.Option("checkpoint"),
    ssc_top_k: int = typer.Option(2),
    lr: float = typer.Option(1e-3),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto"),
) -> None:
    """Run minimal synthetic MC eval smoke loop and print JSON metrics."""
    metrics = run_smoke_eval(
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        segment_size=segment_size,
        aggregation=aggregation,
        segmentation=segmentation,
        state_init_mode=state_init_mode,
        ssc_top_k=ssc_top_k,
        lr=lr,
        seed=seed,
        device=device,
    )
    typer.echo(json.dumps(metrics, sort_keys=True, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
