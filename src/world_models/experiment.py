"""Reproducible run manifests, reused by every chapter that generates rollouts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# The fields that determine the output. Two runs agreeing on all of these
# should produce the same rollout in the same environment.
REPRODUCIBILITY_FIELDS = (
    "checkpoint", "revision", "seed", "num_frames",
    "guidance_scale", "num_inference_steps", "height", "width",
    "prompt", "observation",
)


def _versions(
    packages: tuple[str, ...] = (
        "torch",
        "diffusers",
        "transformers",
        "cosmos_guardrail",
        "accelerate",
    ),
) -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to reproduce and identify one rollout run."""

    checkpoint: str
    revision: str
    seed: int
    num_frames: int
    guidance_scale: float
    num_inference_steps: int
    height: int
    width: int
    prompt: str
    observation: str                      # input clip path or content hash
    outputs: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=_versions)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def run_id(self) -> str:
        """Content-addressed id from the reproducibility fields alone."""
        payload = json.dumps(
            {k: getattr(self, k) for k in REPRODUCIBILITY_FIELDS}, sort_keys=True
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def save(self, run_dir: Path, filename: str = "manifest.json") -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        record = asdict(self) | {"run_id": self.run_id}
        path.write_text(json.dumps(record, indent=2))
        return path
