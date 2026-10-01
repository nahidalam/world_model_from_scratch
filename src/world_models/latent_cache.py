"""Prepare checked video latents using only the frozen, pinned WAN VAE.

The cache stores normalized posterior modes on CPU. Its final cache.json is
written only after every split and reconstruction artifact has been saved.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from world_models.cosmos_generation import CHECKPOINT, REVISION
from world_models.toy_video import GENERATOR_VERSION, SPLITS, VideoSpec, generate_clip, split_seeds


SCHEMA_VERSION = 1
SOURCE = {"checkpoint": CHECKPOINT, "revision": REVISION, "subfolder": "vae"}
_IDENTITY_FIELDS = ("schema_version", "generator", "spec", "source", "source_sha256", "normalization", "splits",
                    "latent_shape", "observed_latent_frames", "storage_dtype")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(metadata: dict) -> str:
    payload = {name: metadata[name] for name in _IDENTITY_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a CUDA GPU for the measured cache experiment, or --device cpu")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("The cache supports cpu and cuda devices")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def _load_vae(device: torch.device):
    # Importing/loading the VAE directly avoids loading the text encoder or DiT.
    from diffusers import AutoencoderKLWan

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    vae = AutoencoderKLWan.from_pretrained(
        CHECKPOINT, subfolder="vae", revision=REVISION, torch_dtype=dtype,
    ).to(device).eval()
    vae.requires_grad_(False)
    return vae


def _normalization(vae) -> dict:
    if vae.config.z_dim != 16:
        raise ValueError("The chapter expects a 16-channel WAN VAE")
    mean = torch.as_tensor(vae.config.latents_mean, dtype=torch.float32)
    std = torch.as_tensor(vae.config.latents_std, dtype=torch.float32)
    if mean.shape != (16,) or std.shape != (16,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("VAE normalization must have sixteen finite means and positive standard deviations")
    return {"mean": mean.tolist(), "std": std.tolist(), "formula": "(posterior_mode - mean) / std"}


def _statistics(normalization: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(normalization["mean"], device=device, dtype=torch.float32).view(1, 16, 1, 1, 1)
    std = torch.tensor(normalization["std"], device=device, dtype=torch.float32).view(1, 16, 1, 1, 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid cached latent normalization")
    return mean, std


def _pixels(frames: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype) / 127.5 - 1


def _uint8(decoded: torch.Tensor) -> np.ndarray:
    return ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1).cpu().numpy()


def _save_reconstruction(output: Path, original: np.ndarray, reconstructed: np.ndarray, fps: int) -> dict:
    import imageio.v3 as iio

    if reconstructed.shape != original.shape:
        raise ValueError(f"VAE reconstruction shape {reconstructed.shape} differs from input {original.shape}")
    paired = np.concatenate((original, reconstructed), axis=2)
    # Input on the left, reconstruction on the right; three times down the PNG.
    indices = (0, len(original) // 2, len(original) - 1)
    iio.imwrite(output / "reconstruction.png", np.concatenate([paired[index] for index in indices], axis=0))
    iio.imwrite(output / "reconstruction.mp4", paired, fps=fps, macro_block_size=1)
    delta = (reconstructed.astype(np.float32) - original.astype(np.float32)) / 255
    mse = float(np.mean(delta ** 2))
    return {"mae_0_1": float(np.mean(np.abs(delta))), "rmse_0_1": float(np.sqrt(mse)),
            "psnr_db": float(-10 * np.log10(mse)) if mse > 0 else None,
            "png": "reconstruction.png", "video": "reconstruction.mp4",
            "layout": "input left; reconstruction right"}


@torch.inference_mode()
def prepare_cache(output: str | Path, device: str | torch.device, train_clips: int = 512,
                  val_clips: int = 64, test_clips: int = 64, spec: VideoSpec = VideoSpec(),
                  seed: int = 0) -> dict:
    """Encode each trajectory once and return the completed cache metadata.

    Output must be absent or empty. Failed runs remain incomplete and cannot
    be consumed as caches. All split seeds denote independent trajectories.
    """
    if not isinstance(spec, VideoSpec):
        raise TypeError("spec must be a VideoSpec")
    seeds_by_split = {name: split_seeds(name, count, seed) for name, count in
                      zip(SPLITS, (train_clips, val_clips, test_clips))}
    target = _device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Cache output must be an empty directory; choose a new output path")
    # Exclusive creation also prevents two preparers from sharing a directory.
    with (output / ".preparing").open("x") as handle:
        handle.write("cache.json is written when preparation is complete\n")
    started = time.perf_counter()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        torch.cuda.reset_peak_memory_stats(target)
    vae = _load_vae(target)
    normalization = _normalization(vae)
    mean, std = _statistics(normalization, target)
    latent_shape = (16, spec.latent_frames, spec.size // 8, spec.size // 8)
    metadata = {"schema_version": SCHEMA_VERSION, "generator": GENERATOR_VERSION,
                "spec": asdict(spec), "source": dict(SOURCE), "normalization": normalization,
                "source_sha256": {name: _sha256(Path(__file__).with_name(name))
                                   for name in ("toy_video.py", "latent_cache.py")},
                "latent_shape": list(latent_shape), "observed_latent_frames": spec.observed_latent_frames,
                "storage_dtype": "float32", "splits": {}, "causality": []}
    encode_started = time.perf_counter()
    for split, seeds in seeds_by_split.items():
        latents = torch.empty((len(seeds), *latent_shape), dtype=torch.float32)
        positions = torch.empty((len(seeds), spec.frames, 2), dtype=torch.float32)
        for index, trajectory_seed in enumerate(seeds):
            clip = generate_clip(spec, int(trajectory_seed))
            pixels = _pixels(clip["frames"], target, vae.dtype)
            raw = vae.encode(pixels).latent_dist.mode()
            clean = (raw.float() - mean) / std
            if tuple(clean.shape) != (1, *latent_shape) or not torch.isfinite(clean).all():
                raise ValueError("VAE produced invalid latent shape or nonfinite values")
            latents[index].copy_(clean[0].cpu())
            positions[index].copy_(torch.from_numpy(clip["positions"]))
            if index == 0:
                prefix = vae.encode(pixels[:, :, :spec.observed_frames]).latent_dist.mode()
                prefix = (prefix.float() - mean) / std
                expected = clean[:, :, :spec.observed_latent_frames]
                passed = prefix.shape == expected.shape and torch.allclose(prefix, expected, rtol=2e-3, atol=2e-3)
                max_error = float((prefix - expected).abs().max()) if prefix.shape == expected.shape else None
                metadata["causality"].append({"split": split, "trajectory_id": clip["trajectory_id"],
                    "max_abs": max_error, "atol": 2e-3, "rtol": 2e-3, "passed": bool(passed)})
                if not passed:
                    raise ValueError(f"VAE causality check failed for {split}: observed latents depend on future frames")
                if split == "train":
                    decoded = vae.decode(raw.to(vae.dtype), return_dict=False)[0]
                    metadata["reconstruction"] = _save_reconstruction(output, clip["frames"], _uint8(decoded)[0], spec.fps)
                    del decoded
                del prefix, expected
            del pixels, raw, clean
            if (index + 1) % 64 == 0 or index + 1 == len(seeds):
                print(f"cache {split}: {index + 1}/{len(seeds)} clips", flush=True)
        path = output / f"{split}.pt"
        pending = output / f".{split}.pt.tmp"
        torch.save({"latents": latents, "seeds": torch.from_numpy(seeds.copy()), "positions": positions}, pending)
        pending.replace(path)
        metadata["splits"][split] = {"file": path.name, "sha256": _sha256(path), "count": len(seeds),
                                     "seed_start": int(seeds[0]), "seed_stop": int(seeds[-1]) + 1}
        del latents, positions
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    metadata.update({"created_at": datetime.now(timezone.utc).isoformat(), "device": str(target),
        "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
        "vae_dtype": str(vae.dtype), "versions": {"torch": str(torch.__version__), "diffusers": version("diffusers")},
        "seconds": time.perf_counter() - started, "encoding_seconds": time.perf_counter() - encode_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(target) if target.type == "cuda" else None})
    del vae
    if target.type == "cuda":
        torch.cuda.empty_cache()
    metadata["artifacts"] = {name: _sha256(output / name) for name in ("reconstruction.png", "reconstruction.mp4")}
    metadata["cache_id"] = _identity(metadata)
    pending = output / ".cache.json.tmp"
    pending.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (output / ".preparing").unlink()
    pending.replace(output / "cache.json")
    return metadata


def read_cache(path: str | Path) -> dict:
    """Validate a completed cache, its identity, and every stored checksum."""
    root = Path(path)
    metadata = json.loads((root / "cache.json").read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("source") != SOURCE:
        raise ValueError("Unsupported cache schema or VAE source")
    spec = VideoSpec(**metadata["spec"])
    if metadata["latent_shape"] != [16, spec.latent_frames, spec.size // 8, spec.size // 8]:
        raise ValueError("Cache latent shape differs from its video specification")
    if metadata["observed_latent_frames"] != spec.observed_latent_frames or metadata["storage_dtype"] != "float32":
        raise ValueError("Cache conditioning boundary or storage dtype is invalid")
    _statistics(metadata["normalization"], torch.device("cpu"))
    if metadata.get("cache_id") != _identity(metadata):
        raise ValueError("Cache identity does not match its metadata")
    if set(metadata["splits"]) != set(SPLITS):
        raise ValueError("Cache must contain train, val, and test splits")
    intervals = []
    for split in SPLITS:
        record = metadata["splits"][split]
        if record["file"] != f"{split}.pt" or record["count"] < 1 or record["seed_stop"] - record["seed_start"] != record["count"]:
            raise ValueError(f"Invalid split metadata: {split}")
        if _sha256(root / record["file"]) != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {split}.pt")
        intervals.append((record["seed_start"], record["seed_stop"]))
    for index, (start, stop) in enumerate(intervals):
        if any(start < other_stop and other_start < stop for other_start, other_stop in intervals[:index]):
            raise ValueError("Cache split seeds overlap")
    for name in ("reconstruction.png", "reconstruction.mp4"):
        if _sha256(root / name) != metadata["artifacts"][name]:
            raise ValueError(f"Checksum mismatch for {name}")
    if len(metadata.get("causality", [])) != 3 or not all(item.get("passed") is True for item in metadata["causality"]):
        raise ValueError("Cache needs passing causality checks for all three splits")
    return metadata


def load_split(path: str | Path, split: str) -> dict[str, torch.Tensor]:
    """Load checked CPU tensors, using PyTorch's restricted weights loader."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    metadata = read_cache(path)
    record = metadata["splits"][split]
    values = torch.load(Path(path) / record["file"], map_location="cpu", weights_only=True)
    if not isinstance(values, dict) or set(values) != {"latents", "seeds", "positions"} or not all(isinstance(v, torch.Tensor) for v in values.values()):
        raise ValueError("Cache split must contain only latents, seeds, and positions tensors")
    expected = {"latents": (record["count"], *metadata["latent_shape"]),
                "seeds": (record["count"],), "positions": (record["count"], metadata["spec"]["frames"], 2)}
    for name, tensor in values.items():
        dtype = torch.int64 if name == "seeds" else torch.float32
        if tuple(tensor.shape) != expected[name] or tensor.dtype != dtype or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid {name} tensor in {split} cache")
    if not torch.equal(values["seeds"], torch.arange(record["seed_start"], record["seed_stop"], dtype=torch.int64)):
        raise ValueError("Cached seeds differ from split metadata")
    if (values["positions"] < 0).any() or (values["positions"] >= metadata["spec"]["size"]).any():
        raise ValueError("Cached positions are outside the frame")
    return values


@torch.inference_mode()
def decode_latents(latents: torch.Tensor, metadata: dict, device: str | torch.device) -> np.ndarray:
    """Load only the pinned VAE, decode one clip at a time, then release it."""
    if metadata.get("source") != SOURCE:
        raise ValueError("Unsupported VAE source")
    if latents.ndim != 5 or latents.shape[0] < 1 or list(latents.shape[1:]) != metadata["latent_shape"] or not torch.isfinite(latents).all():
        raise ValueError("latents must be finite [B,C,T,H,W] tensors matching the cache")
    target = _device(device)
    vae = _load_vae(target)
    if _normalization(vae) != metadata["normalization"]:
        raise ValueError("Loaded VAE normalization differs from the cache")
    mean, std = _statistics(metadata["normalization"], target)
    results = []
    for clip in latents.split(1):
        raw = clip.to(device=target, dtype=torch.float32) * std + mean
        decoded = vae.decode(raw.to(vae.dtype), return_dict=False)[0]
        results.append(_uint8(decoded))
    result = np.concatenate(results)
    spec = VideoSpec(**metadata["spec"])
    if result.shape[1:] != (spec.frames, spec.size, spec.size, 3):
        raise ValueError("Decoded video shape differs from cache specification")
    del vae
    if target.type == "cuda":
        torch.cuda.empty_cache()
    return result
