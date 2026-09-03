"""Cosmos-specific backend: checkpoint metadata, boundary adapters, and the driver.

This is the only module that drives the diffusers Cosmos pipeline. Everything
above it depends on the book's model-agnostic interfaces, not on Cosmos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CheckpointSpec:
    """A checkpoint and the input contract it expects.

    Values are transcribed from the official model card and should be
    re-verified against the loaded pipeline configuration. ``num_context_frames``
    applies to video conditioning; conditioning on a still image uses one frame.
    """

    repo_id: str
    revision: str
    num_context_frames: int
    height: int
    width: int
    fps: int


CHECKPOINTS: dict[str, CheckpointSpec] = {
    "cosmos-predict2.5-2b-720p": CheckpointSpec(
        repo_id="nvidia/Cosmos-Predict2.5-2B",
        revision="diffusers/base/post-trained",
        num_context_frames=5,
        height=704,
        width=1280,
        fps=16,
    ),
}


def apply_guardrail_compatibility_patch() -> None:
    """Keep cosmos_guardrail 0.3.1 active while fixing its device property.

    Version 0.3.1 exposes a read-only property that fails when Diffusers checks
    the pipeline device. Other versions are left unchanged.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        guardrail_version = version("cosmos_guardrail")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "cosmos_guardrail is required for Cosmos inference. Install "
            "requirements-chapter-02-gpu.txt."
        ) from exc

    if guardrail_version == "0.3.1":
        import torch
        from cosmos_guardrail.cosmos_guardrail import CosmosSafetyChecker

        CosmosSafetyChecker.device = property(lambda self: torch.device("cuda"))


def to_conditioning(observation) -> list:
    """Convert a canonical Observation into pipeline conditioning frames.

    Returns the frames as PIL images. The pipeline resizes them to the
    checkpoint resolution and selects the conditioning window itself, so no
    trimming or resizing is done here.
    """
    from PIL import Image

    return [Image.fromarray(frame) for frame in observation.frames]


def to_frames(pipeline_output) -> np.ndarray:
    """Convert a Cosmos pipeline result into (T, H, W, 3) uint8 frames."""
    frames = pipeline_output.frames[0]
    return np.stack([np.asarray(f) for f in frames])


class CosmosBackend:
    """Turns an Observation into generated frames using the Cosmos pipeline.

    The only class that drives Diffusers. Construct it from a loaded pipeline
    and its CheckpointSpec.
    """

    def __init__(self, pipeline, spec: CheckpointSpec):
        self.pipeline = pipeline
        self.spec = spec

    def predict(self, observation, *, prompt: str, seed: int, num_frames: int,
                guidance_scale: float, num_inference_steps: int) -> np.ndarray:
        import torch

        with torch.inference_mode():
            result = self.pipeline(
                image=None,
                video=to_conditioning(observation),
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )
        return to_frames(result)
