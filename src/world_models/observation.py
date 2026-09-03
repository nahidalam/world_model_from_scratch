"""Model-agnostic observation container and loaders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False)
class Observation:
    """A chronological sequence of RGB frames, oldest first.

    Frames are held as ``(T, H, W, 3)`` uint8 in ``[0, 255]``. Conversion to a
    checkpoint's expected format happens at the model boundary, not here.
    """

    frames: np.ndarray
    fps: float | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        validate_frames(self.frames)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def resolution(self) -> tuple[int, int]:
        """Frame size as ``(height, width)``."""
        return int(self.frames.shape[1]), int(self.frames.shape[2])

    def last(self, n: int) -> "Observation":
        """Keep the ``n`` most recent frames, preserving order."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if n > self.num_frames:
            raise ValueError(
                f"Requested {n} frames but observation has {self.num_frames}. "
                "Supply a longer clip or reduce the context length."
            )
        return Observation(self.frames[-n:], fps=self.fps, source=self.source)


def validate_frames(frames: np.ndarray) -> None:
    """Raise if ``frames`` violates the observation contract."""
    if not isinstance(frames, np.ndarray):
        raise TypeError(f"frames must be a numpy array, got {type(frames).__name__}")
    if frames.ndim != 4:
        raise ValueError(
            f"frames must be 4-D (T, H, W, C), got shape {frames.shape}. "
            "A single image should be given a leading time axis of length 1."
        )
    if frames.shape[0] == 0:
        raise ValueError("frames must contain at least one frame")
    if frames.shape[-1] != 3:
        raise ValueError(
            f"frames must have 3 channels (RGB), got {frames.shape[-1]}. "
            "Drop the alpha channel or expand grayscale before constructing."
        )
    if frames.dtype != np.uint8:
        raise TypeError(
            f"frames must be uint8 in [0, 255], got {frames.dtype}. "
            "Normalization belongs at the model boundary, not here."
        )


def load_video(path: str | Path) -> Observation:
    """Load every frame of a video file into an Observation.

    Requires the imageio ffmpeg backend.
    """
    import imageio.v3 as iio

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such video: {path}")

    # index=... stacks all frames along a new leading axis. Omitting it lets
    # the plugin decide, which may yield only the first frame.
    frames = np.asarray(iio.imread(path, index=...))
    if frames.ndim == 3:  # (T, H, W) grayscale video
        frames = frames[..., np.newaxis]
    return Observation(_to_rgb_uint8(frames), source=path)


def load_image(path: str | Path) -> Observation:
    """Load a single image as a one-frame Observation."""
    import imageio.v3 as iio

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such image: {path}")

    frame = np.asarray(iio.imread(path))
    if frame.ndim == 2:  # (H, W) grayscale
        frame = frame[..., np.newaxis]
    return Observation(_to_rgb_uint8(frame[np.newaxis, ...]), source=path)


def _to_rgb_uint8(frames: np.ndarray) -> np.ndarray:
    """Coerce 4-D decoded frames to (T, H, W, 3) uint8, or raise."""
    if frames.ndim != 4:
        raise ValueError(
            f"expected 4-D (T, H, W, C) before conversion, got shape {frames.shape}"
        )

    channels = frames.shape[-1]
    if channels == 1:  # grayscale
        frames = np.repeat(frames, 3, axis=-1)
    elif channels == 4:  # RGBA
        frames = frames[..., :3]
    elif channels != 3:
        raise ValueError(f"unsupported channel count {channels}; expected 1, 3, or 4")

    if frames.dtype != np.uint8:
        raise TypeError(
            f"Decoded frames have dtype {frames.dtype}; expected uint8. "
            "Convert explicitly so the scaling convention is visible."
        )
    return np.ascontiguousarray(frames)
