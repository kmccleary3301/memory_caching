from __future__ import annotations

import argparse
import json
import math
import time
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
    grad_accum_steps: int,
    batch_size: int,
    max_start: int,
) -> None:
    draws = max(0, steps) * max(1, grad_accum_steps)
    for _ in range(draws):
        _ = torch.randint(0, max_start, (batch_size,), generator=generator)


def _lr_for_step(
    *,
    step_index: int,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    schedule: str,
) -> float:
    if total_steps <= 0:
        return base_lr

    if warmup_steps > 0 and step_index < warmup_steps:
        return base_lr * float(step_index + 1) / float(warmup_steps)

    if schedule == "cosine":
        denom = max(1, total_steps - warmup_steps)
        progress = float(step_index - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    return base_lr


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
    scheduler_state: dict[str, Any],
    optim_config: dict[str, Any],
    step_metrics_tail: list[dict[str, Any]],
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
        "step_metrics_tail": step_metrics_tail[-20:],
        "scheduler_state": scheduler_state,
        "optim_config": optim_config,
        "rng_state_torch": torch.get_rng_state(),
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
                "step_metrics_tail": step_metrics_tail[-20:],
                "scheduler_state": scheduler_state,
                "optim_config": optim_config,
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
) -> tuple[int, list[float], dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            model_state = payload.get("model_state")
            optim_state = payload.get("optimizer_state")
            if isinstance(model_state, dict):
                model.load_state_dict(model_state)
            if isinstance(optim_state, dict):
                optimizer.load_state_dict(optim_state)
            rng_state = payload.get("rng_state_torch")
            if isinstance(rng_state, torch.Tensor):
                torch.set_rng_state(rng_state)
            global_step = int(payload.get("global_step", 0))
            loss_tail = payload.get("loss_tail", [])
            losses = [float(x) for x in loss_tail] if isinstance(loss_tail, list) else []
            scheduler_state = payload.get("scheduler_state", {})
            if not isinstance(scheduler_state, dict):
                scheduler_state = {}
            step_metrics = payload.get("step_metrics_tail", [])
            if not isinstance(step_metrics, list):
                step_metrics = []
            return global_step, losses, scheduler_state, step_metrics
    except Exception:
        pass

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"unsupported resume payload format: {path}")
    return int(payload.get("global_step", 0)), [], {}, []


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
    parser.add_argument("--optim-config", default="configs/optim/schedules.yaml")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    optim_cfg_path = Path(args.optim_config)
    optim_cfg = _load_yaml(optim_cfg_path)

    ckpt_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.global_step < 0:
        raise SystemExit("global-step must be >= 0")
    if args.max_steps <= 0:
        raise SystemExit("max-steps must be positive")
    if args.max_seq_len < 8:
        raise SystemExit("max-seq-len must be at least 8")
    if args.grad_accum_steps <= 0:
        raise SystemExit("grad-accum-steps must be positive")
    if args.clip_grad_norm <= 0:
        raise SystemExit("clip-grad-norm must be positive")

    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from is not None and not resume_from.exists():
        raise SystemExit(f"resume checkpoint does not exist: {resume_from}")

    requested_steps = int(cfg.get("steps", 100))
    effective_steps = min(requested_steps, int(args.max_steps))
    run_name = str(cfg.get("name", "train"))
    cfg_seq_len = int(cfg.get("seq_len", 128))
    cfg_batch_size = int(cfg.get("batch_size", 4))
    save_every = max(1, int(cfg.get("save_every", 100)))
    d_model = int(cfg.get("d_model", 64))
    vocab_size = int(cfg.get("vocab_size", 32768))
    effective_seq_len = min(cfg_seq_len, int(args.max_seq_len))

    schedule = str(optim_cfg.get("schedule", "cosine")).strip().lower()
    base_lr = float(cfg.get("lr", optim_cfg.get("base_lr", 1e-3)))
    warmup_steps = int(optim_cfg.get("warmup_steps", 0))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    betas = optim_cfg.get("betas", [0.9, 0.95])
    eps = float(optim_cfg.get("eps", 1e-8))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(float(betas[0]), float(betas[1])),
        eps=eps,
    )

    scaler: torch.cuda.amp.GradScaler | None = None
    if args.amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    global_step = int(args.global_step)
    losses: list[float] = []
    step_metrics: list[dict[str, Any]] = []
    scheduler_state: dict[str, Any] = {
        "schedule": schedule,
        "base_lr": base_lr,
        "warmup_steps": warmup_steps,
        "total_steps": effective_steps,
        "last_step": global_step,
    }

    if resume_from is not None:
        global_step, losses, scheduler_state_loaded, step_metrics_loaded = _restore_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
        )
        scheduler_state.update(scheduler_state_loaded)
        step_metrics.extend(step_metrics_loaded)

    _advance_sampler(
        generator=generator,
        steps=global_step,
        grad_accum_steps=args.grad_accum_steps,
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
        scheduler_state=scheduler_state,
        optim_config={
            "path": str(optim_cfg_path),
            "schedule": schedule,
            "base_lr": base_lr,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "betas": [float(betas[0]), float(betas[1])],
            "eps": eps,
            "grad_accum_steps": args.grad_accum_steps,
            "clip_grad_norm": args.clip_grad_norm,
            "amp": bool(args.amp and device.type == "cuda"),
            "deterministic": bool(args.deterministic),
        },
        step_metrics_tail=step_metrics,
    )

    model.train()
    loss_start = float("nan")
    loss_end = float("nan")
    step_times: list[float] = []
    tokens_per_s: list[float] = []

    for local_idx in range(effective_steps):
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            inp, tgt = _sample_batch(
                stream=stream,
                seq_len=effective_seq_len,
                batch_size=cfg_batch_size,
                generator=generator,
            )
            inp = inp.to(device)
            tgt = tgt.to(device)

            use_amp = bool(scaler is not None)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(inp)
                    raw_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)
                    )
                    loss = raw_loss / float(args.grad_accum_steps)
                scaler.scale(loss).backward()
                accum_loss += float(raw_loss.item())
            else:
                logits = model(inp)
                raw_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)
                )
                loss = raw_loss / float(args.grad_accum_steps)
                loss.backward()
                accum_loss += float(raw_loss.item())

        current_lr = _lr_for_step(
            step_index=global_step,
            base_lr=base_lr,
            total_steps=effective_steps,
            warmup_steps=warmup_steps,
            schedule=schedule,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

        loss_value = accum_loss / float(args.grad_accum_steps)
        if local_idx == 0:
            loss_start = loss_value
        loss_end = loss_value
        losses.append(loss_value)
        global_step += 1
        scheduler_state["last_step"] = global_step

        duration = max(time.perf_counter() - start_time, 1e-12)
        step_times.append(duration)
        step_token_count = cfg_batch_size * effective_seq_len * args.grad_accum_steps
        tps = float(step_token_count / duration)
        tokens_per_s.append(tps)
        step_metrics.append(
            {
                "step": global_step,
                "loss": loss_value,
                "lr": current_lr,
                "step_time_s": duration,
                "tokens_per_s": tps,
            }
        )

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
                scheduler_state=scheduler_state,
                optim_config={
                    "path": str(optim_cfg_path),
                    "schedule": schedule,
                    "base_lr": base_lr,
                    "warmup_steps": warmup_steps,
                    "weight_decay": weight_decay,
                    "betas": [float(betas[0]), float(betas[1])],
                    "eps": eps,
                    "grad_accum_steps": args.grad_accum_steps,
                    "clip_grad_norm": args.clip_grad_norm,
                    "amp": bool(args.amp and device.type == "cuda"),
                    "deterministic": bool(args.deterministic),
                },
                step_metrics_tail=step_metrics,
            )

    metrics_path = ckpt_dir / "train_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in step_metrics) + "\n"
    )

    max_memory_mb = 0.0
    if device.type == "cuda":
        max_memory_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024**2))

    meta = {
        "stage": "train",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "optim_config": str(optim_cfg_path),
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
        "base_lr": base_lr,
        "schedule": schedule,
        "warmup_steps": warmup_steps,
        "grad_accum_steps": args.grad_accum_steps,
        "clip_grad_norm": args.clip_grad_norm,
        "amp": bool(args.amp and device.type == "cuda"),
        "deterministic": bool(args.deterministic),
        "device": str(device),
        "save_every": save_every,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_delta": (loss_start - loss_end) if losses else 0.0,
        "loss_tail": losses[-20:],
        "metrics_file": str(metrics_path),
        "avg_step_time_s": float(sum(step_times) / len(step_times)) if step_times else 0.0,
        "avg_tokens_per_s": float(sum(tokens_per_s) / len(tokens_per_s)) if tokens_per_s else 0.0,
        "max_memory_mb": max_memory_mb,
        "checkpoint_files": [str(p) for p in sorted(ckpt_dir.glob("step_*.pt"))],
    }
    Path("artifacts/train_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/train_manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
