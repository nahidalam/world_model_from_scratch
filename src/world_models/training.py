"""Single-device flow-matching training with reproducible optimizer checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from world_models.flow import flow_matching_loss, make_noisy_latents, replace_conditioned
from world_models.small_world import SmallWorldModel, autocast_context, prefix_mask, small_world_config


@dataclass(frozen=True)
class TrainConfig:
    preset: str = "base"
    steps: int = 2000
    batch_size: int = 4
    accumulation: int = 8
    learning_rate: float = 3e-4
    seed: int = 17
    precision: str = "auto"
    overfit_clips: int = 0
    checkpoint_blocks: bool = False
    evaluate_every: int = 100
    save_every: int = 100

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "accumulation", "evaluate_every", "save_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or self.overfit_clips < 0:
            raise ValueError("Use a positive finite learning rate and a nonnegative overfit_clips count.")
        if self.preset not in ("tiny", "small", "base") or self.precision not in ("auto", "fp32", "bf16", "fp16"):
            raise ValueError("Choose a documented preset and precision.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in (
        "training.py", "small_world.py", "flow.py", "models/cosmos_transformer.py")}


def write_json(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_device_precision(device: str, precision: str) -> tuple[torch.device, str]:
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("The chapter supports cpu for component checks and cuda for video training.")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this profile. Run the CPU smoke check here, then train on a CUDA host.")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(target)
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if target.type == "cuda" else "fp32"
    with autocast_context(target, precision):
        pass
    return target, precision


def runtime_record(device: torch.device) -> dict:
    return {"torch": str(torch.__version__), "python": platform.python_version(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
            "cuda": torch.version.cuda}


def memory_record(device: torch.device) -> dict:
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def _loss(model: SmallWorldModel, clean: Tensor, observed_frames: int, rng: torch.Generator,
          device: torch.device, precision: str) -> Tensor:
    clean = clean.to(device=device, dtype=torch.float32)
    noise = torch.randn(clean.shape, generator=rng).to(device)
    levels = torch.rand(clean.shape[0], generator=rng).to(device)
    noisy, target = make_noisy_latents(clean, noise, levels[:, None, None, None, None])
    mask = prefix_mask(clean, observed_frames)
    with autocast_context(device, precision):
        prediction = model(replace_conditioned(noisy, clean, mask), levels, mask)
        return flow_matching_loss(prediction, target, mask)


@torch.no_grad()
def validation_loss(model: SmallWorldModel, latents: Tensor, observed_frames: int,
                    device: torch.device, precision: str, batch_size: int = 4) -> float:
    """Fix validation noise independently of the training random stream."""
    was_training = model.training
    model.eval()
    rng = torch.Generator().manual_seed(739)
    total = 0.0
    try:
        for start in range(0, len(latents), batch_size):
            batch = latents[start:start + batch_size]
            total += float(_loss(model, batch, observed_frames, rng, device, precision)) * len(batch)
        return total / len(latents)
    finally:
        model.train(was_training)


def load_training_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise ValueError("Expected a Chapter 4 training checkpoint with schema_version=1.")
    return checkpoint


def train_latents(train: Tensor, validation: Tensor, metadata: dict, output: Path,
                  config: TrainConfig = TrainConfig(), *, device: str = "cpu",
                  resume: Path | None = None) -> dict:
    """Optimize cached normalized latents; steps counts accumulated updates.

    Checkpoints are written at optimizer boundaries and include the data/noise
    generator, global RNG state, optimizer, scaler, model, and cache identity.
    Increasing ``steps`` is the only configuration change accepted on resume.
    """
    target, precision = resolve_device_precision(device, config.precision)
    for name, data in (("train", train), ("validation", validation)):
        if data.ndim != 5 or len(data) < 1 or not data.is_floating_point() or not torch.isfinite(data).all():
            raise ValueError(f"{name} must contain finite floating [N,C,T,H,W] latents.")
    if train.shape[1:] != validation.shape[1:] or list(train.shape[1:]) != metadata["latent_shape"]:
        raise ValueError("Training, validation, and cache metadata must share a latent shape.")
    observed_frames = int(metadata["observed_latent_frames"])
    prefix_mask(train[:1], observed_frames)
    if config.overfit_clips > len(train):
        raise ValueError("overfit_clips must fit within the training split.")
    if config.overfit_clips:
        train = train[:config.overfit_clips]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if resume is None or Path(resume).resolve() != (output / "checkpoint.pt").resolve():
            raise FileExistsError("Choose a fresh output directory or resume its checkpoint.pt.")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = load_training_checkpoint(resume) if resume is not None else None
    effective_config = asdict(config) | {"precision": precision}
    if checkpoint is not None:
        if checkpoint["cache_id"] != metadata["cache_id"]:
            raise ValueError("The checkpoint and latent cache must have the same identity.")
        if {k: v for k, v in checkpoint["train_config"].items() if k != "steps"} != {
                k: v for k, v in effective_config.items() if k != "steps"}:
            raise ValueError("Resume preserves the training configuration; only steps may increase.")
        if config.steps <= checkpoint["step"]:
            raise ValueError("Set steps to a total greater than the saved optimizer step.")
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    torch.manual_seed(config.seed)
    model = SmallWorldModel(small_world_config(config.preset, tuple(train.shape[1:]))).to(target)
    model.transformer.gradient_checkpointing = config.checkpoint_blocks
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(target.type == "cuda" and precision == "fp16"))
    rng = torch.Generator().manual_seed(config.seed + 1)
    start_step = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        rng.set_state(checkpoint["train_rng"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        if target.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], target)
        start_step = checkpoint["step"]
    # The fixed probes describe both overfitting and held-out velocity prediction.
    train_probe = train[:min(8, len(train))]
    initial_train = validation_loss(model, train_probe, observed_frames, target, precision, config.batch_size)
    initial_validation = validation_loss(model, validation, observed_frames, target, precision, config.batch_size)
    log_path = output / "metrics.jsonl"
    if checkpoint is not None and log_path.exists():
        rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= start_step))
    cumulative_seconds = checkpoint.get("cumulative_training_seconds", 0.0) if checkpoint else 0.0
    skipped_updates = checkpoint.get("skipped_updates", 0) if checkpoint else 0
    source = source_identity()
    started = time.perf_counter()
    step_durations: list[float] = []

    def save(step: int, elapsed: float) -> None:
        payload = {
            "schema_version": 1, "step": step, "model": model.state_dict(),
            "model_config": model.transformer.config_dict(), "train_config": effective_config,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "train_rng": rng.get_state(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(target) if target.type == "cuda" else None,
            "cache_id": metadata["cache_id"], "cache_metadata": metadata,
            "source": source, "runtime": runtime_record(target),
            "cumulative_training_seconds": cumulative_seconds + elapsed,
            "skipped_updates": skipped_updates,
        }
        temporary = output / "checkpoint.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(output / "checkpoint.pt")

    with log_path.open("a") as log:
        for step in range(start_step + 1, config.steps + 1):
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(config.accumulation):
                indices = torch.randint(len(train), (config.batch_size,), generator=rng)
                loss = _loss(model, train[indices], observed_frames, rng, target, precision)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"The training loss is nonfinite at step {step}.")
                scaler.scale(loss / config.accumulation).backward()
                loss_sum += float(loss.detach())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                 error_if_nonfinite=not scaler.is_enabled())
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied = scaler.get_scale() >= old_scale
            skipped_updates += int(not applied)
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            duration = time.perf_counter() - step_started
            step_durations.append(duration)
            row = {"step": step, "loss": loss_sum / config.accumulation,
                   "gradient_norm": float(norm) if torch.isfinite(norm) else None,
                   "optimizer_step_applied": applied, "seconds": duration}
            if step % config.evaluate_every == 0 or step == config.steps:
                row["validation_loss"] = validation_loss(model, validation, observed_frames, target, precision,
                                                         config.batch_size)
                print(f"step {step}: loss={row['loss']:.5f}, validation={row['validation_loss']:.5f}, {duration:.3f}s/update", flush=True)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if step % config.save_every == 0 or step == config.steps:
                save(step, time.perf_counter() - started)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elapsed = time.perf_counter() - started
    final_train = validation_loss(model, train_probe, observed_frames, target, precision, config.batch_size)
    final_validation = validation_loss(model, validation, observed_frames, target, precision, config.batch_size)
    # Discard early timing samples; memory peaks still include all updates and optimizer allocation.
    timings = step_durations[min(5, max(0, len(step_durations) - 1)):]
    mean_seconds = sum(timings) / len(timings)
    params = sum(p.numel() for p in model.parameters())
    report = {
        "schema_version": 1, "experiment": "cached_latent_training", "cache_id": metadata["cache_id"],
        "config": effective_config, "model_config": model.transformer.config_dict(),
        "parameters": params, "parameter_state_estimate_bytes": params * 16,
        "effective_batch_size": config.batch_size * config.accumulation,
        "train_examples": len(train), "validation_examples": len(validation),
        "start_step": start_step, "end_step": config.steps, "skipped_updates_total": skipped_updates,
        "initial_train_probe_loss": initial_train, "final_train_probe_loss": final_train,
        "initial_validation_loss": initial_validation, "final_validation_loss": final_validation,
        "segment_seconds": elapsed, "cumulative_training_seconds": cumulative_seconds + elapsed,
        "mean_update_seconds_after_warmup": mean_seconds,
        "examples_per_second_after_warmup": config.batch_size * config.accumulation / mean_seconds,
        "checkpoint_sha256": file_sha256(output / "checkpoint.pt"),
        "runtime": runtime_record(target), "source": source, **memory_record(target),
        "scope": "Velocity-loss and optimizer measurements. Decoded held-out rollouts are evaluated separately.",
    }
    write_json(output / "training.json", report)
    return report
