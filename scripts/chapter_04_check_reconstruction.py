#!/usr/bin/env python3
"""Measure frozen-VAE reconstruction and bucket detection on excavator clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from world_models import latent_cache, small_world_evaluation, toy_video
from world_models.latent_cache import _device, _save_reconstruction, decode_latents, load_split, read_cache
from world_models.small_world_evaluation import TRACKING, trajectory_metrics
from world_models.toy_video import VideoSpec, generate_clip
from world_models.training import file_sha256, memory_record, runtime_record, write_json


def check_reconstruction(data: Path, output: Path, *, device: str = "cuda", split: str = "val",
                         examples: int = 8) -> dict:
    """Decode ground-truth latents and apply the same future metrics as generation."""
    if split not in ("train", "val", "test") or examples < 1:
        raise ValueError("Choose a dataset split and a positive example count.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh reconstruction output directory.")
    metadata = read_cache(data)
    generator_sha256 = file_sha256(Path(toy_video.__file__))
    if metadata["source_sha256"]["toy_video.py"] != generator_sha256:
        raise ValueError("Use the generator source recorded in the cache to reconstruct evaluation targets.")
    payload = load_split(data, split)
    if examples > len(payload["latents"]):
        raise ValueError("examples must fit within the selected split.")
    target = _device(device)
    output.mkdir(parents=True, exist_ok=True)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    decoded = decode_latents(payload["latents"][:examples], metadata, target)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    decoding = {"seconds": time.perf_counter() - started, **memory_record(target)}
    spec = VideoSpec(**metadata["spec"])
    rows = []
    for index, frames in enumerate(decoded):
        trajectory_seed = int(payload["seeds"][index])
        original = generate_clip(spec, trajectory_seed)
        directory = output / f"trajectory_{trajectory_seed}"
        directory.mkdir()
        reconstruction = _save_reconstruction(directory, original["frames"], frames, spec.fps)
        metrics = trajectory_metrics(frames, original["frames"], original["positions"], spec.observed_frames)
        # Include the step from the last observation into the predicted future.
        future_positions = original["positions"][spec.observed_frames - 1:]
        path_length = np.linalg.norm(np.diff(future_positions, axis=0), axis=1).sum()
        displacement = np.linalg.norm(future_positions[-1] - future_positions[0])
        row = {"trajectory_seed": trajectory_seed, "trajectory_id": original["trajectory_id"],
               "future_path_length_px": float(path_length),
               "future_displacement_px": float(displacement), "metrics": metrics,
               "reconstruction": reconstruction, "directory": directory.name}
        write_json(directory / "metrics.json", row)
        rows.append(row)
    aggregate = {key: float(np.mean([row["metrics"][key] for row in rows])) for key in (
        "future_pixel_rmse", "object_detection_fraction", "centroid_error_penalized_px",
        "final_centroid_error_penalized_px")}
    decoder_dtype = torch.bfloat16 if target.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    report = {
        "schema_version": 1,
        "experiment": "cached_ground_truth_reconstruction",
        "cache_id": metadata["cache_id"],
        "generator": metadata["generator"],
        "tracking": dict(TRACKING),
        "data": str(Path(data).resolve()),
        "split": split,
        "examples": examples,
        "observed_frames": spec.observed_frames,
        "decoder_dtype": str(decoder_dtype),
        "vae_source": metadata["source"],
        "aggregate": aggregate,
        "trajectories": rows,
        "decoding": decoding,
        "seconds": time.perf_counter() - started,
        "runtime": runtime_record(target),
        "source": {
            "scripts/chapter_04_check_reconstruction.py": file_sha256(Path(__file__)),
            "latent_cache.py": file_sha256(Path(latent_cache.__file__)),
            "small_world_evaluation.py": file_sha256(Path(small_world_evaluation.__file__)),
            "toy_video.py": generator_sha256,
        },
        "scope": "Frozen-VAE reconstruction of ground-truth cached latents. Future-only pixel and "
                 "bucket-center metrics establish the compression and detector reference. Paired media "
                 "show ground truth on the left and VAE reconstruction on the right. Future path length "
                 "and displacement describe bucket motion from the last observed frame.",
    }
    write_json(output / "reconstruction_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")
    torch.set_num_threads(4)
    print(json.dumps(check_reconstruction(args.data, args.output, device=args.device,
                                         split=args.split, examples=args.examples), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
