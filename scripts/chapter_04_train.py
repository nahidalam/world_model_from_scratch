#!/usr/bin/env python3
"""Prepare videos, train a small world model, and evaluate saved checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from world_models.training import TrainConfig, file_sha256, load_training_checkpoint, train_latents, write_json


def preview_world(output: Path, seed: int) -> dict:
    """Render the dataset's rule and observation boundary before loading a VAE."""
    from dataclasses import asdict
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw
    from world_models.toy_video import VideoSpec, generate_clip
    from world_models.training import file_sha256
    from world_models import toy_video

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh preview output directory.")
    output.mkdir(parents=True, exist_ok=True)
    spec = VideoSpec()
    clip = generate_clip(spec, seed)
    selected = [0, 4, 8, 12, 16]
    tile = 192
    canvas = Image.new("RGB", (len(selected) * tile, tile + 52), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    for col, index in enumerate(selected):
        frame = Image.fromarray(clip["frames"][index]).resize((tile, tile), Image.Resampling.NEAREST)
        canvas.paste(frame, (col * tile, 52))
        label = "Observed" if index < spec.observed_frames else "Future target"
        color = "#2369a7" if index < spec.observed_frames else "#b34721"
        draw.text((col * tile + 12, 10), f"Frame {index}", fill="#14243b")
        draw.text((col * tile + 12, 29), label, fill=color)
    canvas.save(output / "preview.png")
    iio.imwrite(output / "trajectory.mp4", clip["frames"], fps=spec.fps, macro_block_size=1)
    record = {"experiment": "procedural_dataset_preview", "spec": asdict(spec), "seed": seed,
              "trajectory_id": clip["trajectory_id"], "positions": clip["positions"].tolist(),
              "source_sha256": file_sha256(Path(toy_video.__file__)),
              "frames_sha256": hashlib.sha256(np.ascontiguousarray(clip["frames"]).tobytes()).hexdigest(),
              "scope": "Ground-truth frames from the procedural world used to create the training dataset."}
    write_json(output / "preview.json", record)
    return record


def cpu_smoke(output: Path, device: str) -> dict:
    """Exercise real optimizer updates and exact resume on generated latent tensors."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh smoke output directory.")
    rng = torch.Generator().manual_seed(21)
    train = torch.randn(8, 4, 3, 4, 4, generator=rng) * 0.1
    validation = torch.randn(4, 4, 3, 4, 4, generator=rng) * 0.1
    metadata = {"cache_id": hashlib.sha256(train.numpy().tobytes() + validation.numpy().tobytes()).hexdigest(),
                "latent_shape": list(train.shape[1:]), "observed_latent_frames": 1,
                "scope": "Synthetic latent tensors for optimizer checks."}
    common = dict(preset="tiny", batch_size=2, accumulation=2, learning_rate=0.002,
                  precision="fp32", evaluate_every=20, save_every=20)
    full = train_latents(train, validation, metadata, output / "continuous", TrainConfig(steps=40, **common), device=device)
    train_latents(train, validation, metadata, output / "resumed", TrainConfig(steps=20, **common), device=device)
    train_latents(train, validation, metadata, output / "resumed", TrainConfig(steps=40, **common),
                  device=device, resume=output / "resumed/checkpoint.pt")
    first = load_training_checkpoint(output / "continuous/checkpoint.pt")
    second = load_training_checkpoint(output / "resumed/checkpoint.pt")
    max_error = max(float((first["model"][key] - second["model"][key]).abs().max()) for key in first["model"])
    if max_error != 0 or not torch.equal(first["train_rng"], second["train_rng"]):
        raise AssertionError("Resumed training must reproduce the continuous optimizer trajectory.")
    if full["final_train_probe_loss"] >= full["initial_train_probe_loss"]:
        raise AssertionError("The smoke experiment must reduce its fixed training-probe loss.")
    report = {"experiment": "optimizer_and_resume_smoke", "steps": 40, "parameters": full["parameters"],
              "initial_loss": full["initial_train_probe_loss"], "final_loss": full["final_train_probe_loss"],
              "resume_parameter_max_error": max_error, "runtime": full["runtime"],
              "source": first["source"] | {"scripts/chapter_04_train.py": file_sha256(Path(__file__))},
              "scope": "Optimizer, loss, and exact resume on synthetic latent tensors. Video learning and consumer GPU memory are separate checks."}
    write_json(output / "smoke.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for reproducible local checks.")
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="Render a ground-truth trajectory on CPU.")
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--seed", type=int, default=0)
    prepare = commands.add_parser("prepare", help="Render trajectories and cache frozen WAN VAE latents.")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--train-clips", type=int, default=512)
    prepare.add_argument("--val-clips", type=int, default=64)
    prepare.add_argument("--test-clips", type=int, default=64)
    prepare.add_argument("--size", type=int, default=128)
    prepare.add_argument("--frames", type=int, default=17)
    prepare.add_argument("--observed-frames", type=int, default=5)
    prepare.add_argument("--fps", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=0)
    train = commands.add_parser("train", help="Train with AdamW and save a resumable checkpoint.")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--preset", choices=("tiny", "small", "base"), default="base")
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--accumulation", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    train.add_argument("--overfit-clips", type=int, default=0)
    train.add_argument("--checkpoint-blocks", action="store_true")
    train.add_argument("--evaluate-every", type=int, default=100)
    train.add_argument("--save-every", type=int, default=100)
    train.add_argument("--resume", type=Path)
    sample = commands.add_parser("sample", help="Decode generated futures and evaluate excavator bucket trajectories.")
    sample.add_argument("--data", type=Path, required=True)
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--device", default="cuda")
    sample.add_argument("--split", choices=("train", "val", "test"), default="test")
    sample.add_argument("--examples", type=int, default=8)
    sample.add_argument("--sampling-steps", type=int, default=30)
    sample.add_argument("--seed", type=int, default=123)
    sample.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    smoke = commands.add_parser("smoke", help="Run optimizer and exact-resume checks on small synthetic tensors.")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "preview":
        result = preview_world(args.output, args.seed)
    elif args.command == "prepare":
        from world_models.latent_cache import prepare_cache
        from world_models.toy_video import VideoSpec
        result = prepare_cache(args.output, args.device, train_clips=args.train_clips,
                               val_clips=args.val_clips, test_clips=args.test_clips,
                               spec=VideoSpec(size=args.size, frames=args.frames,
                                              observed_frames=args.observed_frames, fps=args.fps), seed=args.seed)
    elif args.command == "train":
        from world_models.latent_cache import load_split, read_cache
        metadata = read_cache(args.data)
        config = TrainConfig(**{key: getattr(args, key) for key in TrainConfig.__dataclass_fields__})
        result = train_latents(load_split(args.data, "train")["latents"], load_split(args.data, "val")["latents"],
                               metadata, args.output, config, device=args.device, resume=args.resume)
    elif args.command == "sample":
        from world_models.small_world_evaluation import evaluate_checkpoint
        result = evaluate_checkpoint(args.data, args.checkpoint, args.output, device=args.device,
                                     split=args.split, examples=args.examples, sampling_steps=args.sampling_steps,
                                     seed=args.seed, precision=args.precision)
    else:
        result = cpu_smoke(args.output, args.device)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
