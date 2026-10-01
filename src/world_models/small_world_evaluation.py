"""Evaluate excavator bucket motion against observed-frame persistence."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from world_models.experiment import RunManifest
from world_models.models.cosmos_transformer import CosmosTransformerConfig
from world_models.small_world import SmallWorldModel, sample_future
from world_models.training import (file_sha256, load_training_checkpoint, memory_record,
                                   resolve_device_precision, runtime_record, source_identity, write_json)


TRACKING = {
    "target": "bucket",
    "method": "largest eight-connected red-excess region; color-weighted centroid",
    "red_excess_threshold": 40,
    "minimum_reference_area_fraction": 0.25,
    "maximum_reference_area_fraction": 4.0,
    "missing_detection_penalty": "image diagonal in pixels",
}


def _red_weights(frame: np.ndarray) -> np.ndarray:
    pixels = frame.astype(np.float32)
    return np.maximum(pixels[..., 0] - np.maximum(pixels[..., 1], pixels[..., 2])
                      - TRACKING["red_excess_threshold"], 0)


def _largest_region(weights: np.ndarray) -> np.ndarray:
    """Keep one connected silhouette so isolated red noise cannot move its center."""
    remaining = weights > 0
    height, width = remaining.shape
    largest = []
    for y, x in zip(*np.nonzero(remaining)):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        pending, region = [(int(y), int(x))], []
        while pending:
            row, col = pending.pop()
            region.append((row, col))
            for other_row in range(max(0, row - 1), min(height, row + 2)):
                for other_col in range(max(0, col - 1), min(width, col + 2)):
                    if remaining[other_row, other_col]:
                        remaining[other_row, other_col] = False
                        pending.append((other_row, other_col))
        if len(region) > len(largest):
            largest = region
    selected = np.zeros_like(weights)
    if largest:
        rows, cols = np.asarray(largest).T
        selected[rows, cols] = weights[rows, cols]
    return selected


def trajectory_metrics(predicted: np.ndarray, truth: np.ndarray, positions: np.ndarray,
                       observed_frames: int) -> dict:
    """Measure future pixels and the excavator bucket's rendered center.

    Select the largest eight-connected region with R - max(G, B) > 40.
    Its area must be 0.25x to 4x the ground-truth bucket silhouette's area;
    this reference follows the bucket's rendered size and orientation.
    Missing detections receive an image-diagonal distance. These position
    estimates accompany videos for inspection of the bucket and arm shape.
    """
    if predicted.shape != truth.shape or truth.ndim != 4 or truth.shape[-1] != 3:
        raise ValueError("predicted and truth must have matching [T,H,W,3] shapes.")
    if positions.shape != (len(truth), 2) or not 0 < observed_frames < len(truth):
        raise ValueError("Supply one xy position per frame and a valid observed prefix.")
    pred = predicted[observed_frames:].astype(np.float32)
    target = truth[observed_frames:].astype(np.float32)
    yy, xx = np.indices(truth.shape[1:3], dtype=np.float32)
    distances, detected, estimates = [], [], []
    diagonal = float(math.hypot(truth.shape[1], truth.shape[2]))
    for frame, reference, position in zip(pred, target, positions[observed_frames:]):
        weights = _largest_region(_red_weights(frame))
        reference_area = np.count_nonzero(_red_weights(reference))
        area = np.count_nonzero(weights)
        visible = (reference_area > 0
                   and TRACKING["minimum_reference_area_fraction"] * reference_area <= area
                   <= TRACKING["maximum_reference_area_fraction"] * reference_area)
        if visible:
            center = np.array([(weights * xx).sum(), (weights * yy).sum()]) / weights.sum()
            distance = float(np.linalg.norm(center - position))
            estimates.append(center.tolist())
        else:
            distance = diagonal
            estimates.append(None)
        detected.append(bool(visible))
        distances.append(distance)
    visible_errors = [error for error, visible in zip(distances, detected) if visible]
    return {"future_pixel_rmse": float(np.sqrt(np.mean(((pred - target) / 255.0) ** 2))),
            "object_detection_fraction": float(np.mean(detected)),
            "centroid_error_detected_px": float(np.mean(visible_errors)) if visible_errors else None,
            "centroid_error_penalized_px": float(np.mean(distances)),
            "final_centroid_error_penalized_px": distances[-1],
            "final_frame_detected": detected[-1],
            "future_frames": len(pred), "estimated_centers": estimates,
            "per_frame_detected": detected, "per_frame_penalized_error_px": distances}


def comparison_image(path: Path, truth: np.ndarray, predicted: np.ndarray,
                     persistence: np.ndarray, observed_frames: int) -> None:
    indices = np.linspace(observed_frames, len(truth) - 1, 4, dtype=int)
    height, width = truth.shape[1:3]
    image = Image.new("RGB", (4 * width, 3 * (height + 24)), "white")
    draw = ImageDraw.Draw(image)
    for row, (label, frames) in enumerate((("Ground truth", truth), ("Generated", predicted), ("Repeat last frame", persistence))):
        for col, index in enumerate(indices):
            x, y = col * width, row * (height + 24)
            draw.text((x + 3, y + 5), f"{label}: {index}", fill="black")
            image.paste(Image.fromarray(frames[index]), (x, y + 24))
    image.save(path)


def evaluate_checkpoint(data: Path, checkpoint_path: Path, output: Path, *, device: str = "cuda",
                        split: str = "test", examples: int = 8, sampling_steps: int = 30,
                        seed: int = 123, precision: str = "auto") -> dict:
    from world_models.latent_cache import decode_latents, load_split, read_cache
    from world_models.toy_video import VideoSpec, generate_clip
    import imageio.v3 as iio

    if split not in ("train", "val", "test") or examples < 1 or sampling_steps < 1:
        raise ValueError("Choose a dataset split and positive example and sampling-step counts.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh evaluation output directory.")
    target, resolved_precision = resolve_device_precision(device, precision)
    metadata = read_cache(data)
    from world_models import latent_cache, toy_video
    if metadata["source_sha256"]["toy_video.py"] != file_sha256(Path(toy_video.__file__)):
        raise ValueError("Use the generator source recorded in the cache to reconstruct evaluation targets.")
    payload = load_split(data, split)
    if examples > len(payload["latents"]):
        raise ValueError("examples must fit within the selected split.")
    checkpoint = load_training_checkpoint(checkpoint_path)
    if checkpoint["cache_id"] != metadata["cache_id"]:
        raise ValueError("Evaluate the checkpoint with the cache used for training.")
    output.mkdir(parents=True, exist_ok=True)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    model = SmallWorldModel(CosmosTransformerConfig(**checkpoint["model_config"])).to(target).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    observed_count = metadata["observed_latent_frames"]
    generated = []
    for index in range(examples):
        prefix = payload["latents"][index:index + 1, :, :observed_count]
        sample = sample_future(model, prefix, total_frames=metadata["latent_shape"][1],
                               steps=sampling_steps, seed=seed + index, precision=resolved_precision).cpu()
        torch.testing.assert_close(sample[:, :, :observed_count], prefix.float(), rtol=0, atol=0)
        generated.append(sample)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    sampling = {"seconds": time.perf_counter() - started, **memory_record(target)}
    del model, checkpoint
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    latent_batch = torch.cat(generated)
    decoded = decode_latents(latent_batch, metadata, target)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    decoding = {"seconds": time.perf_counter() - started, **memory_record(target)}
    spec = VideoSpec(**metadata["spec"])
    checkpoint_hash = file_sha256(checkpoint_path)
    source = source_identity() | {"small_world_evaluation.py": file_sha256(Path(__file__)),
                                  "latent_cache.py": file_sha256(Path(latent_cache.__file__)),
                                  "toy_video.py": file_sha256(Path(toy_video.__file__))}
    decoder_dtype = torch.bfloat16 if target.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    inference = {"source": source, "sampling_precision": resolved_precision,
                 "decoder_dtype": str(decoder_dtype), "vae_source": metadata["source"],
                 "cache_id": metadata["cache_id"], "solver": "uniform-euler-v1"}
    implementation_hash = hashlib.sha256(json.dumps(inference, sort_keys=True).encode()).hexdigest()
    rows = []
    for index, frames in enumerate(decoded):
        trajectory_seed = int(payload["seeds"][index])
        original = generate_clip(spec, trajectory_seed)
        truth = original["frames"]
        persistence = truth.copy()
        persistence[spec.observed_frames:] = truth[spec.observed_frames - 1]
        clip_dir = output / f"trajectory_{trajectory_seed}"
        clip_dir.mkdir()
        for filename, video in (("rollout.mp4", frames), ("ground_truth.mp4", truth), ("persistence.mp4", persistence)):
            iio.imwrite(clip_dir / filename, video, fps=spec.fps, macro_block_size=1)
        np.save(clip_dir / "frames.npy", frames)
        torch.save(latent_batch[index].clone(), clip_dir / "latents.pt")
        comparison_image(clip_dir / "comparison.png", truth, frames, persistence, spec.observed_frames)
        prefix_hash = hashlib.sha256(truth[:spec.observed_frames].tobytes()).hexdigest()
        manifest = RunManifest(
            checkpoint=str(Path(checkpoint_path).resolve()), revision=checkpoint_hash,
            seed=seed + index, num_frames=spec.frames, guidance_scale=1.0,
            num_inference_steps=sampling_steps, height=spec.size, width=spec.size, prompt="",
            observation=f"{metadata['cache_id']}/{split}/{trajectory_seed}",
            conditioning_frames_sha256=prefix_hash, conditioning_num_frames=spec.observed_frames,
            conditioning_fps=spec.fps, output_fps=spec.fps,
            implementation="chapter04-small-world-euler-v1", implementation_sha256=implementation_hash,
            outputs=["rollout.mp4", "frames.npy", "latents.pt", "comparison.png", "inference.json"],
        )
        write_json(clip_dir / "inference.json", inference)
        manifest.save(clip_dir)
        row = {"trajectory_seed": trajectory_seed, "run_id": manifest.run_id,
               "generated": trajectory_metrics(frames, truth, original["positions"], spec.observed_frames),
               "persistence": trajectory_metrics(persistence, truth, original["positions"], spec.observed_frames)}
        write_json(clip_dir / "metrics.json", row)
        rows.append(row)
    aggregates = {label: {key: float(np.mean([row[label][key] for row in rows])) for key in (
        "future_pixel_rmse", "object_detection_fraction", "centroid_error_penalized_px",
        "final_centroid_error_penalized_px")}
        for label in ("generated", "persistence")}
    report = {"schema_version": 1, "experiment": "decoded_trajectory_evaluation", "cache_id": metadata["cache_id"],
              "generator": metadata["generator"], "tracking": dict(TRACKING),
              "checkpoint_sha256": checkpoint_hash, "split": split, "examples": examples,
              "sampling_steps": sampling_steps, "seed": seed, "precision": resolved_precision,
              "observed_latent_max_error": 0.0, "aggregate": aggregates, "trajectories": rows,
              "sampling": sampling, "decoding": decoding, "runtime": runtime_record(target),
              "source": source, "inference": inference,
              "scope": "Future-only pixel and bucket-center metrics for the generated excavator scene. "
                       "Missing detections receive image-diagonal error. Inspect comparison images and "
                       "videos for bucket and arm shape; these metrics measure motion prediction."}
    write_json(output / "evaluation.json", report)
    return report
