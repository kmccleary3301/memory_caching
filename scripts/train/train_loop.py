from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.norm(x)
        return self.head(x)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return loaded


def _load_token_stream(data_dir: Path) -> list[int]:
    shards = sorted(data_dir.glob("shard_*.jsonl"))
    tokens: list[int] = []
    for shard in shards:
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            ids = payload.get("token_ids")
            if isinstance(ids, list):
                for value in ids:
                    if isinstance(value, int):
                        tokens.append(value)
    return tokens


def _build_synthetic_tokens(length: int, vocab_size: int, seed: int) -> list[int]:
    g = torch.Generator()
    g.manual_seed(seed)
    out = torch.randint(0, vocab_size, (length,), generator=g, dtype=torch.long)
    return [int(x) for x in out.tolist()]


def _sample_batch(
    *,
    stream: list[int],
    seq_len: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(stream) - (seq_len + 1)
    if max_start <= 0:
        raise SystemExit("token stream too short for requested seq_len")

    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    inp = torch.empty(batch_size, seq_len, dtype=torch.long)
    tgt = torch.empty(batch_size, seq_len, dtype=torch.long)
    for row_idx, st in enumerate(starts.tolist()):
        segment = stream[st : st + seq_len + 1]
        inp[row_idx] = torch.tensor(segment[:-1], dtype=torch.long)
        tgt[row_idx] = torch.tensor(segment[1:], dtype=torch.long)
    return inp, tgt


def _advance_sampler(
    *,
    generator: torch.Generator,
    steps: int,
    batch_size: int,
    max_start: int,
) -> None:
    for _ in range(steps):
        _ = torch.randint(0, max_start, (batch_size,), generator=generator)


def _save_checkpoint(
    *,
    path: Path,
    model: TinyLM,
    optimizer: torch.optim.Optimizer,
    config_path: Path,
    run_name: str,
    global_step: int,
    sequence_length: int,
    batch_size: int,
    seed: int,
    losses: list[float],
) -> None:
    payload = {
        "schema_version": "v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "global_step": global_step,
        "config_path": str(config_path),
        "backend_type": "tiny_lm",
        "aggregation_type": "n/a",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "seed": seed,
        "git_commit": "unknown",
        "loss_tail": losses[-20:],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "timestamp_utc": payload["timestamp_utc"],
                "run_name": run_name,
                "global_step": global_step,
                "config_path": str(config_path),
                "backend_type": "tiny_lm",
                "aggregation_type": "n/a",
                "sequence_length": sequence_length,
                "batch_size": batch_size,
                "seed": seed,
                "git_commit": "unknown",
                "checkpoint_file": str(path),
                "loss_tail": losses[-20:],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _restore_checkpoint(
    *,
    path: Path,
    model: TinyLM,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, list[float]]:
    try:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            model_state = payload.get("model_state")
            optim_state = payload.get("optimizer_state")
            if isinstance(model_state, dict):
                model.load_state_dict(model_state)
            if isinstance(optim_state, dict):
                optimizer.load_state_dict(optim_state)
            global_step = int(payload.get("global_step", 0))
            loss_tail = payload.get("loss_tail", [])
            losses = [float(x) for x in loss_tail] if isinstance(loss_tail, list) else []
            return global_step, losses
    except Exception:
        pass

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"unsupported resume payload format: {path}")
    return int(payload.get("global_step", 0)), []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    ckpt_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.global_step < 0:
        raise SystemExit("global-step must be >= 0")
    if args.max_steps <= 0:
        raise SystemExit("max-steps must be positive")
    if args.max_seq_len < 8:
        raise SystemExit("max-seq-len must be at least 8")

    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from is not None and not resume_from.exists():
        raise SystemExit(f"resume checkpoint does not exist: {resume_from}")

    requested_steps = int(cfg.get("steps", 100))
    effective_steps = min(requested_steps, int(args.max_steps))
    run_name = str(cfg.get("name", "train"))
    cfg_seq_len = int(cfg.get("seq_len", 128))
    cfg_batch_size = int(cfg.get("batch_size", 4))
    save_every = max(1, int(cfg.get("save_every", 100)))
    lr = float(cfg.get("lr", 1e-3))
    d_model = int(cfg.get("d_model", 64))
    vocab_size = int(cfg.get("vocab_size", 32768))
    effective_seq_len = min(cfg_seq_len, int(args.max_seq_len))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    stream = _load_token_stream(data_dir)
    if len(stream) < effective_seq_len + 2:
        stream = _build_synthetic_tokens(
            length=max(8192, effective_seq_len * 16),
            vocab_size=vocab_size,
            seed=args.seed,
        )
    max_start = len(stream) - (effective_seq_len + 1)
    if max_start <= 0:
        raise SystemExit("token stream too short for requested seq_len")

    stream_max = max(stream) if len(stream) > 0 else 0
    dynamic_vocab_size = max(vocab_size, stream_max + 2)
    model = TinyLM(vocab_size=dynamic_vocab_size, d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = int(args.global_step)
    losses: list[float] = []
    if resume_from is not None:
        global_step, losses = _restore_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
        )
    _advance_sampler(
        generator=generator,
        steps=global_step,
        batch_size=cfg_batch_size,
        max_start=max_start,
    )

    initial_ckpt = ckpt_dir / f"step_{global_step:06d}.pt"
    _save_checkpoint(
        path=initial_ckpt,
        model=model,
        optimizer=optimizer,
        config_path=cfg_path,
        run_name=run_name,
        global_step=global_step,
        sequence_length=effective_seq_len,
        batch_size=cfg_batch_size,
        seed=args.seed,
        losses=losses,
    )

    model.train()
    loss_start = float("nan")
    loss_end = float("nan")
    for local_idx in range(effective_steps):
        inp, tgt = _sample_batch(
            stream=stream,
            seq_len=effective_seq_len,
            batch_size=cfg_batch_size,
            generator=generator,
        )
        inp = inp.to(device)
        tgt = tgt.to(device)

        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = float(loss.item())
        if local_idx == 0:
            loss_start = loss_value
        loss_end = loss_value
        losses.append(loss_value)
        global_step += 1

        should_save = (global_step % save_every == 0) or (local_idx == effective_steps - 1)
        if should_save:
            _save_checkpoint(
                path=ckpt_dir / f"step_{global_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                config_path=cfg_path,
                run_name=run_name,
                global_step=global_step,
                sequence_length=effective_seq_len,
                batch_size=cfg_batch_size,
                seed=args.seed,
                losses=losses,
            )

    metrics_path = ckpt_dir / "train_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            json.dumps({"step": idx, "loss": float(val)}, sort_keys=True)
            for idx, val in enumerate(losses, start=1)
        )
        + ("\n" if losses else "")
    )

    meta = {
        "stage": "train",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "data_dir": args.data_dir,
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_file": str(ckpt_dir / f"step_{global_step:06d}.pt"),
        "global_step": global_step,
        "seed": args.seed,
        "resume_from": str(resume_from) if resume_from is not None else None,
        "run_name": run_name,
        "requested_steps": requested_steps,
        "executed_steps": effective_steps,
        "sequence_length": effective_seq_len,
        "batch_size": cfg_batch_size,
        "vocab_size": dynamic_vocab_size,
        "lr": lr,
        "device": str(device),
        "save_every": save_every,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_delta": (loss_start - loss_end) if losses else 0.0,
        "loss_tail": losses[-20:],
        "metrics_file": str(metrics_path),
        "checkpoint_files": [str(p) for p in sorted(ckpt_dir.glob("step_*.pt"))],
    }
    Path("artifacts/train_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/train_manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
