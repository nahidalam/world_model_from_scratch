"""RolloutEngine: assembles a backend, generation, and a reproducible manifest."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from world_models.observation import Observation
from world_models.experiment import RunManifest


class WorldModelBackend(Protocol):
    """A model that predicts frames from an observation.

    `spec` exposes repo_id, revision, height, width, and fps.
    """

    spec: object

    def predict(self, observation: Observation, *, prompt: str, seed: int,
                num_frames: int, guidance_scale: float,
                num_inference_steps: int) -> np.ndarray: ...


@dataclass(frozen=True)
class RolloutResult:
    frames: np.ndarray          # (T, H, W, 3) uint8
    manifest: RunManifest


class RolloutEngine:
    """Generate reproducible rollouts from any backend."""

    def __init__(self, backend: WorldModelBackend):
        self.backend = backend

    def generate(self, observation: Observation, *, prompt: str, seed: int,
                 num_frames: int = 93, guidance_scale: float = 7.0,
                 num_inference_steps: int = 36,
                 save_to: Path | None = None) -> RolloutResult:
        spec = self.backend.spec
        frames = self.backend.predict(
            observation, prompt=prompt, seed=seed, num_frames=num_frames,
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
        )
        manifest = RunManifest(
            checkpoint=spec.repo_id, revision=spec.revision, seed=seed,
            num_frames=num_frames, guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=spec.height, width=spec.width, prompt=prompt,
            observation=str(observation.source) if observation.source else "in-memory",
        )
        if save_to is not None:
            manifest = self._save(frames, manifest, Path(save_to))
        return RolloutResult(frames=frames, manifest=manifest)

    def _save(self, frames: np.ndarray, manifest: RunManifest, root: Path) -> RunManifest:
        import imageio.v3 as iio

        run_dir = root / manifest.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        video = run_dir / "rollout.mp4"
        iio.imwrite(str(video), frames, fps=self.backend.spec.fps)
        manifest = replace(manifest, outputs=[str(video)])
        manifest.save(run_dir)
        return manifest
