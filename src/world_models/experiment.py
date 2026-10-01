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

# Omitted values preserve the identifiers of manifests shipped before Chapter 3.
OPTIONAL_REPRODUCIBILITY_FIELDS = (
    "conditioning_frames_sha256", "conditioning_num_frames",
    "conditioning_fps", "output_fps",
    "implementation", "implementation_sha256",
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
    conditioning_frames_sha256: str | None = None
    conditioning_num_frames: int | None = None
    conditioning_fps: float | None = None
    output_fps: float | None = None
    implementation: str | None = None
    implementation_sha256: str | None = None

    @property
    def run_id(self) -> str:
        """Content-addressed id from the reproducibility fields alone."""
        fields = {k: getattr(self, k) for k in REPRODUCIBILITY_FIELDS}
        fields.update({k: getattr(self, k) for k in OPTIONAL_REPRODUCIBILITY_FIELDS
                       if getattr(self, k) is not None})
        payload = json.dumps(fields, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        """Read a saved manifest and reject a mismatched content identifier.

        Legacy records may omit the new conditioning fields. A record without
        a saved run_id is also accepted; its identifier is computed normally.
        """
        record = json.loads(Path(path).read_text())
        if not isinstance(record, dict):
            raise ValueError(f"Manifest must be a JSON object: {path}")
        saved_id = record.pop("run_id", None)
        manifest = cls(**record)
        if saved_id is not None and saved_id != manifest.run_id:
            raise ValueError(f"Manifest run_id mismatch: {path}")
        return manifest

    def save(self, run_dir: Path, filename: str = "manifest.json") -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        record = asdict(self) | {"run_id": self.run_id}
        path.write_text(json.dumps(record, indent=2))
        return path
