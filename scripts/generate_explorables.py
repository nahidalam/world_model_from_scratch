#!/usr/bin/env python3
"""Generate the Chapter 2 explorable assets and their run manifests.

Produces, skipping anything that already exists:
- 4-seed rollout grid (seeds 0-3 at the base settings)
- guidance_scale sweep (3/7/12, seed 0)
- num_inference_steps sweep (20/36/50, seed 0)
- per-frame disagreement map (.npy, not committed) and its heatmap video

Every rollout gets a RunManifest JSON next to the video, named
<video_stem>.manifest.json, so the explorer's assets stay identifiable
and reproducible. Manifests are backfilled even when a video is skipped.

Note on seeding: rollouts were generated with a CUDA-device torch.Generator,
which draws a different random stream than the CPU generator of the same seed.
Reproducing these exact videos requires the device generator used here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from world_models.backends.cosmos import (
    CHECKPOINTS,
    CheckpointSpec,
    apply_guardrail_compatibility_patch,
    to_conditioning,
)
from world_models.experiment import RunManifest
from world_models.observation import load_video
from world_models.visualize import disagreement_map

CHECKPOINT_KEY = "cosmos-predict2.5-2b-720p"
OBSERVATION_PATH = Path("assets/chapter_02/sand_mining.mp4")
PROMPT = (
    "an aerial view of a sand mining operation; "
    "the machinery keeps moving and the water keeps flowing"
)
NUM_FRAMES = 29
BASE_GUIDANCE = 7.0
BASE_STEPS = 15

SEEDS_DIR = Path("assets/chapter_02/seeds")
GUIDANCE_DIR = Path("assets/chapter_02/guidance_sweep")
STEPS_DIR = Path("assets/chapter_02/steps_sweep")
DISAGREEMENT_NPY = Path("assets/chapter_02/disagreement_map.npy")
DISAGREEMENT_MP4 = Path("assets/chapter_02/disagreement_map.mp4")

_pipeline = None
_conditioning = None


def _get_pipeline(spec: CheckpointSpec):
    """Load the pipeline on first use so manifest-only runs stay cheap."""
    global _pipeline
    if _pipeline is None:
        import torch
        from diffusers import Cosmos2_5_PredictBasePipeline

        apply_guardrail_compatibility_patch()

        print(f"Loading pipeline from {spec.repo_id}...")
        _pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
            spec.repo_id,
            revision=spec.revision,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    return _pipeline


def _get_conditioning(spec: CheckpointSpec) -> list:
    global _conditioning
    if _conditioning is None:
        observation = load_video(OBSERVATION_PATH)
        _conditioning = to_conditioning(observation.last(spec.num_context_frames))
    return _conditioning


def ensure_rollout(
    path: Path,
    spec: CheckpointSpec,
    *,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
) -> None:
    """Generate the rollout if missing, then write its manifest if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"  {path.name}: already exists")
    else:
        import torch
        from diffusers.utils import export_to_video

        print(f"  {path.name}: generating...")
        pipeline = _get_pipeline(spec)
        result = pipeline(
            image=None,
            video=_get_conditioning(spec),
            prompt=PROMPT,
            num_frames=NUM_FRAMES,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        export_to_video(result.frames[0], str(path), fps=spec.fps)
        print(f"    saved to {path}")

    manifest_name = f"{path.stem}.manifest.json"
    if (path.parent / manifest_name).exists():
        return
    manifest = RunManifest(
        checkpoint=spec.repo_id,
        revision=spec.revision,
        seed=seed,
        num_frames=NUM_FRAMES,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=spec.height,
        width=spec.width,
        prompt=PROMPT,
        observation=str(OBSERVATION_PATH),
        outputs=[str(path)],
    )
    manifest_path = manifest.save(path.parent, filename=manifest_name)
    print(f"    manifest {manifest.run_id} -> {manifest_path}")


def ensure_disagreement_assets(spec: CheckpointSpec) -> None:
    """Compute the per-frame disagreement map and render it as a video."""
    if DISAGREEMENT_NPY.exists():
        print(f"  {DISAGREEMENT_NPY.name}: already exists")
        disagreement = np.load(DISAGREEMENT_NPY)
    else:
        print(f"  {DISAGREEMENT_NPY.name}: computing...")
        futures = [
            load_video(SEEDS_DIR / f"rollout_seed{seed}.mp4").frames
            for seed in range(4)
        ]
        disagreement = disagreement_map(futures)
        np.save(DISAGREEMENT_NPY, disagreement)
        print(f"    shape {disagreement.shape}, max {disagreement.max():.3f}")

    if DISAGREEMENT_MP4.exists():
        print(f"  {DISAGREEMENT_MP4.name}: already exists")
        return
    print(f"  {DISAGREEMENT_MP4.name}: rendering...")
    from diffusers.utils import export_to_video
    from matplotlib import colormaps
    from PIL import Image

    # Normalize by the volume's max so the brightest disagreement saturates.
    normalized = disagreement / disagreement.max()
    colored = (colormaps["magma"](normalized)[..., :3] * 255).astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in colored]
    export_to_video(frames, str(DISAGREEMENT_MP4), fps=spec.fps)
    print(f"    saved to {DISAGREEMENT_MP4}")


def generate_explorables() -> None:
    spec = CHECKPOINTS[CHECKPOINT_KEY]

    print("\n1. 4-seed baseline...")
    for seed in range(4):
        ensure_rollout(
            SEEDS_DIR / f"rollout_seed{seed}.mp4",
            spec,
            seed=seed,
            guidance_scale=BASE_GUIDANCE,
            num_inference_steps=BASE_STEPS,
        )

    print("\n2. guidance_scale sweep...")
    for guidance in (3.0, 7.0, 12.0):
        ensure_rollout(
            GUIDANCE_DIR / f"guidance_{guidance:g}_seed0.mp4",
            spec,
            seed=0,
            guidance_scale=guidance,
            num_inference_steps=BASE_STEPS,
        )

    print("\n3. num_inference_steps sweep...")
    for steps in (20, 36, 50):
        ensure_rollout(
            STEPS_DIR / f"steps_{steps}_seed0.mp4",
            spec,
            seed=0,
            guidance_scale=BASE_GUIDANCE,
            num_inference_steps=steps,
        )

    print("\n4. Disagreement map...")
    ensure_disagreement_assets(spec)

    print("\nDone!")


if __name__ == "__main__":
    generate_explorables()
