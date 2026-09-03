#!/usr/bin/env python3
"""Generate Chapter 2 rollouts and figures on a CUDA machine."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_PROMPT = (
    "an aerial view of a sand mining operation; "
    "the machinery keeps moving and the water keeps flowing"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chapter 2 seed rollouts and visualization artifacts."
    )
    parser.add_argument("--input", default="assets/chapter_02/sand_mining.mp4")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--checkpoint", default="cosmos-predict2.5-2b-720p")
    parser.add_argument("--num-frames", type=int, default=29)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--num-inference-steps", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--heatmap-frame", type=int, default=20)
    parser.add_argument("--runs-dir", default="outputs/chapter_02/runs")
    parser.add_argument("--seed-rollouts-dir", default="outputs/chapter_02/seeds")
    parser.add_argument("--shipped-seed-rollouts-dir", default="assets/chapter_02/seeds")
    parser.add_argument("--figures-dir", default="outputs/chapter_02/figures")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--make-sweeps", action="store_true")
    return parser.parse_args()


def log(*parts: object) -> None:
    print("[" + time.strftime("%H:%M:%S") + "]", *parts, flush=True)


def save_filmstrip(frames, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from world_models.visualize import sample_frames

    strip = sample_frames(frames, k=6)
    fig, axes = plt.subplots(1, len(strip), figsize=(3 * len(strip), 3))
    for ax, frame in zip(axes, strip):
        ax.imshow(frame)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison(first, second, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from world_models.visualize import sample_frames

    strip0 = sample_frames(first, k=6)
    strip1 = sample_frames(second, k=6)
    fig, axes = plt.subplots(2, strip0.shape[0], figsize=(3 * strip0.shape[0], 6))
    for col, frame in enumerate(strip0):
        axes[0, col].imshow(frame)
        axes[0, col].axis("off")
    for col, frame in enumerate(strip1):
        axes[1, col].imshow(frame)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("seed 0")
    axes[1, 0].set_ylabel("seed 1")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(disagreement, frame_index: int, path: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = min(frame_index, disagreement.shape[0] - 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(disagreement[t], cmap="magma")
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return t


def main() -> None:
    args = parse_args()

    if args.hf_home is not None:
        os.environ.setdefault("HF_HOME", str(Path(args.hf_home).expanduser()))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch
    from diffusers import Cosmos2_5_PredictBasePipeline

    from world_models.backends.cosmos import (
        CHECKPOINTS,
        CosmosBackend,
        apply_guardrail_compatibility_patch,
    )
    from world_models.engine import RolloutEngine
    from world_models.observation import load_video
    from world_models.visualize import disagreement_map

    if len(args.seeds) < 2:
        raise ValueError("need at least two seeds for comparison figures")

    apply_guardrail_compatibility_patch()

    spec = CHECKPOINTS[args.checkpoint]
    input_path = (REPO_ROOT / args.input).resolve()
    runs_dir = (REPO_ROOT / args.runs_dir).resolve()
    seed_rollouts_dir = (REPO_ROOT / args.seed_rollouts_dir).resolve()
    shipped_seed_rollouts_dir = (REPO_ROOT / args.shipped_seed_rollouts_dir).resolve()
    figures_dir = (REPO_ROOT / args.figures_dir).resolve()
    seed_rollouts_dir.mkdir(parents=True, exist_ok=True)
    shipped_seed_rollouts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    log("loading", spec.repo_id, spec.revision)
    started = time.time()
    pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
        spec.repo_id,
        revision=spec.revision,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    log("pipeline loaded in %.1fs" % (time.time() - started))

    engine = RolloutEngine(CosmosBackend(pipeline, spec))
    observation = load_video(input_path)
    context = observation.last(spec.num_context_frames)
    log("observation", observation.frames.shape, "context", context.frames.shape)

    futures = []
    for seed in args.seeds:
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        result = engine.generate(
            context,
            prompt=args.prompt,
            seed=seed,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            save_to=runs_dir,
        )
        rollout_path = seed_rollouts_dir / f"rollout_seed{seed}.mp4"
        shutil.copy2(result.manifest.outputs[0], rollout_path)
        shipped_rollout_path = shipped_seed_rollouts_dir / f"rollout_seed{seed}.mp4"
        shutil.copy2(rollout_path, shipped_rollout_path)
        futures.append(result)
        log(
            "seed", seed,
            "run_id", result.manifest.run_id,
            "frames", result.frames.shape,
            "seconds", "%.1f" % (time.time() - started),
            "peak_vram_gb", "%.2f" % (torch.cuda.max_memory_allocated() / 1e9),
            "rollout", rollout_path,
            "shipped_copy", shipped_rollout_path,
        )

    save_filmstrip(futures[0].frames, figures_dir / "filmstrip_seed0.png")
    save_comparison(
        futures[0].frames,
        futures[1].frames,
        figures_dir / "futures_seed0_vs_seed1.png",
    )

    disagreement = disagreement_map([future.frames for future in futures])
    np.save(figures_dir / "disagreement.npy", disagreement)
    frame_index = save_heatmap(
        disagreement,
        args.heatmap_frame,
        figures_dir / f"disagreement_t{args.heatmap_frame}.png",
    )
    log(
        "disagreement",
        disagreement.shape,
        "mean", "%.4f" % float(disagreement.mean()),
        "max", "%.4f" % float(disagreement.max()),
        "frame", frame_index,
    )

    if args.make_sweeps:
        sweep_specs = [
            (
                REPO_ROOT / "outputs/chapter_02/guidance",
                ((3.0, args.num_frames, args.num_inference_steps),
                 (7.0, args.num_frames, args.num_inference_steps),
                 (12.0, args.num_frames, args.num_inference_steps)),
                lambda guidance, num_frames, steps: f"guidance_{guidance:g}.mp4",
            ),
            (
                REPO_ROOT / "outputs/chapter_02/steps",
                ((7.0, args.num_frames, 15),
                 (7.0, args.num_frames, 20),
                 (7.0, args.num_frames, 36),
                 (7.0, args.num_frames, 50)),
                lambda guidance, num_frames, steps: f"steps_{steps}.mp4",
            ),
        ]
        for out_dir, combos, name_fn in sweep_specs:
            out_dir.mkdir(parents=True, exist_ok=True)
            for guidance, num_frames, steps in combos:
                result = engine.generate(
                    context,
                    prompt=args.prompt,
                    seed=0,
                    num_frames=num_frames,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    save_to=runs_dir,
                )
                target = out_dir / name_fn(guidance, num_frames, steps)
                shutil.copy2(result.manifest.outputs[0], target)
                log("sweep", target)


if __name__ == "__main__":
    main()
