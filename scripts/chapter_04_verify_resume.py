#!/usr/bin/env python3
"""Verify exact optimizer continuation with deterministic kernels on a latent cache.

This utility uses math SDPA, deterministic algorithms, and disabled TF32 while
retaining the requested autocast precision. Its timing and memory describe this
verification profile; the ordinary training command selects its own kernels.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from world_models.latent_cache import load_split, read_cache
from world_models.training import (
    TrainConfig,
    file_sha256,
    load_training_checkpoint,
    train_latents,
    write_json,
)


def configure_determinism() -> dict:
    """Configure this verification process before the first CUDA operation."""
    workspace = ":4096:8"
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace:
        raise RuntimeError("Run verification in a fresh process so cuBLAS can initialize deterministically.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "sdpa_backend": "math",
    }


def compare_state(first, second) -> dict:
    """Compare a nested checkpoint state, including tensor dtype and shape."""
    exact = True
    maximum = 0.0
    comparable = True
    tensor_count = 0
    differences: list[str] = []

    def mismatch(path: str) -> None:
        nonlocal exact
        exact = False
        # Keep a useful diagnosis without expanding a large checkpoint report.
        if len(differences) < 20:
            differences.append(path or "<root>")

    def visit(left, right, path: str) -> None:
        nonlocal maximum, comparable, tensor_count
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            tensor_count += 1
            if left.shape != right.shape or left.dtype != right.dtype:
                comparable = False
                mismatch(path)
                return
            if not torch.equal(left, right):
                mismatch(path)
                if left.numel():
                    delta = float((left.to(torch.float64) - right.to(torch.float64)).abs().max())
                    if math.isfinite(delta):
                        maximum = max(maximum, delta)
                    else:
                        comparable = False
            return
        if type(left) is not type(right):
            comparable = False
            mismatch(path)
            return
        if isinstance(left, dict):
            if left.keys() != right.keys():
                comparable = False
                mismatch(path + ".keys")
            for key in left.keys() & right.keys():
                visit(left[key], right[key], f"{path}.{key}" if path else str(key))
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                comparable = False
                mismatch(path + ".length")
            for index, (value, other) in enumerate(zip(left, right)):
                visit(value, other, f"{path}[{index}]")
        elif left != right:
            mismatch(path)
            if isinstance(left, (int, float, bool)):
                delta = abs(float(left) - float(right))
                if math.isfinite(delta):
                    maximum = max(maximum, delta)
                else:
                    comparable = False
            else:
                comparable = False

    visit(first, second, "")
    return {"exact": exact, "max_abs": maximum if comparable else None,
            "tensor_count": tensor_count, "first_differences": differences}


def verify_resume(data: Path, output: Path, config: TrainConfig, *, device: str = "cuda") -> dict:
    """Compare N continuous updates with N//2 updates followed by a restart."""
    if config.steps < 2:
        raise ValueError("Use at least two steps to include an update after resume.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh resume-verification output directory.")
    determinism = configure_determinism()
    metadata = read_cache(data)
    train = load_split(data, "train")["latents"]
    validation = load_split(data, "val")["latents"]
    split_step = config.steps // 2
    continuous_path = output / "continuous"
    resumed_path = output / "resumed"
    # Math SDPA avoids fused-kernel backward reduction differences while keeping
    # BF16/FP16 autocast active when requested by the training configuration.
    with sdpa_kernel(SDPBackend.MATH):
        continuous_report = train_latents(train, validation, metadata, continuous_path, config, device=device)
        train_latents(train, validation, metadata, resumed_path, replace(config, steps=split_step), device=device)
        shutil.copyfile(resumed_path / "training.json", resumed_path / "training_initial.json")
        split_checkpoint_sha256 = file_sha256(resumed_path / "checkpoint.pt")
        resumed_report = train_latents(train, validation, metadata, resumed_path, config,
                                      device=device, resume=resumed_path / "checkpoint.pt")
    continuous = load_training_checkpoint(continuous_path / "checkpoint.pt")
    resumed = load_training_checkpoint(resumed_path / "checkpoint.pt")
    states = {name: compare_state(continuous[name], resumed[name]) for name in (
        "model", "optimizer", "scaler", "step", "train_rng", "cpu_rng", "cuda_rng", "skipped_updates")}
    exact = all(value["exact"] for value in states.values())
    report = {
        "schema_version": 1,
        "experiment": "cached_latent_resume_verification",
        "cache_id": metadata["cache_id"],
        "data": str(Path(data).resolve()),
        "config": continuous["train_config"],
        "precision": continuous["train_config"]["precision"],
        "determinism": determinism,
        "split_step": split_step,
        "end_step": config.steps,
        "exact": exact,
        "states": states,
        "checkpoint_sha256": {
            "continuous": file_sha256(continuous_path / "checkpoint.pt"),
            "split": split_checkpoint_sha256,
            "resumed": file_sha256(resumed_path / "checkpoint.pt"),
        },
        "runtime": continuous_report["runtime"],
        "source": continuous["source"] | {"scripts/chapter_04_verify_resume.py": file_sha256(Path(__file__))},
        "continuous_training": continuous_report,
        "resumed_training": resumed_report,
        "scope": "Exact model, optimizer, scaler, progress, and random-state continuation on the recorded "
                 "latent cache, device, precision, and deterministic math-SDPA execution settings. "
                 "This verification profile has its own timing and memory measurements. "
                 "Checkpoint file hashes also include runtime "
                 "measurements, so equality is established from the saved states.",
    }
    write_json(output / "resume_verification.json", report)
    if not exact:
        changed = ", ".join(name for name, result in states.items() if not result["exact"])
        raise AssertionError(f"Resume states differ: {changed}. See {output / 'resume_verification.json'}.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset", choices=("small", "base"), default="small")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least two")
    torch.set_num_threads(4)
    config = TrainConfig(preset=args.preset, steps=args.steps, batch_size=args.batch_size,
                         accumulation=args.accumulation, precision=args.precision)
    print(json.dumps(verify_resume(args.data, args.output, config, device=args.device),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
