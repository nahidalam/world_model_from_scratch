"""Deterministic excavator videos with ground-truth bucket positions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


GENERATOR_VERSION = "excavator-arm-v1"
SPLIT_STRIDE = 1_000_000
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class VideoSpec:
    size: int = 128
    frames: int = 17
    observed_frames: int = 5
    fps: int = 8

    def __post_init__(self) -> None:
        for name in ("size", "frames", "observed_frames", "fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.size < 32 or self.size % 16:
            raise ValueError("size must be at least 32 and divisible by 16")
        if self.frames < 5 or (self.frames - 1) % 4:
            raise ValueError("frames must be at least five and have the form 4k+1")
        if self.observed_frames >= self.frames or (self.observed_frames - 1) % 4:
            raise ValueError("observed_frames must have the form 4k+1 and leave future frames")

    @property
    def latent_frames(self) -> int:
        return (self.frames - 1) // 4 + 1

    @property
    def observed_latent_frames(self) -> int:
        return (self.observed_frames - 1) // 4 + 1


def split_seeds(split: str, count: int, seed: int = 0) -> np.ndarray:
    """Give each split its own million-seed interval, independent of its size."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= SPLIT_STRIDE:
        raise ValueError(f"count must be between 1 and {SPLIT_STRIDE}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 3 * SPLIT_STRIDE:
        raise ValueError("seed must be a nonnegative int64 with room for three split intervals")
    start = seed + SPLITS.index(split) * SPLIT_STRIDE
    return np.arange(start, start + count, dtype=np.int64)


def _scene(size: int) -> Image.Image:
    """Draw the fixed camera view at four times the requested resolution."""
    scale = size * 4
    image = Image.new("RGB", (scale, scale), (177, 202, 217))
    draw = ImageDraw.Draw(image)

    def points(values):
        return [(x * scale, y * scale) for x, y in values]

    # Sand and machinery use neutral or yellow tones. Only the bucket is red.
    draw.polygon(points([(0, .78), (.17, .74), (.39, .79), (.62, .75),
                         (.84, .79), (1, .76), (1, 1), (0, 1)]), fill=(182, 165, 126))
    draw.rectangle((0, .86 * scale, scale, scale), fill=(207, 188, 145))
    outline = (43, 52, 56)
    draw.rounded_rectangle((.065 * scale, .76 * scale, .405 * scale, .86 * scale),
                           radius=.045 * scale, fill=outline)
    draw.rounded_rectangle((.095 * scale, .78 * scale, .375 * scale, .84 * scale),
                           radius=.025 * scale, fill=(89, 99, 99))
    for x in np.linspace(.13, .34, 5):
        draw.ellipse(((x - .021) * scale, .789 * scale,
                      (x + .021) * scale, .831 * scale), fill=(51, 62, 65))
    draw.polygon(points([(.09, .66), (.29, .66), (.36, .71), (.36, .77), (.09, .77)]),
                 fill=(212, 184, 66), outline=outline, width=max(1, round(scale * .008)))
    draw.polygon(points([(.13, .50), (.25, .50), (.29, .66), (.13, .66)]),
                 fill=(216, 186, 66), outline=outline, width=max(1, round(scale * .008)))
    draw.polygon(points([(.15, .52), (.233, .52), (.26, .625), (.15, .625)]),
                 fill=(70, 111, 131))
    draw.line(points([(.195, .52), (.195, .625)]), fill=outline,
              width=max(1, round(scale * .008)))
    return image


def _render_frame(scene: Image.Image, joints: np.ndarray, bucket_angle: float,
                  size: int) -> np.ndarray:
    image = scene.copy()
    draw = ImageDraw.Draw(image)
    scale = image.width
    outline, arm = (43, 52, 56), (212, 184, 66)
    points = [(float(x * scale), float(y * scale)) for x, y in joints]
    # Constant link lengths keep the mechanism connected throughout the cycle.
    for start, end, width in ((points[0], points[1], .045), (points[1], points[2], .033)):
        draw.line((start, end), fill=outline, width=max(1, round(scale * (width + .014))))
        draw.line((start, end), fill=arm, width=max(1, round(scale * width)))
    for x, y in points:
        radius = .019 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=outline)
        radius = .010 * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(132, 143, 145))
    # A solid scoop silhouette remains easy to identify after video compression.
    bucket = np.array([[-.035, -.012], [.12, .008], [.105, .086],
                       [.035, .10], [-.025, .058]], dtype=np.float64)
    cosine, sine = np.cos(bucket_angle), np.sin(bucket_angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    bucket = bucket @ rotation.T + joints[-1]
    draw.polygon([(float(x * scale), float(y * scale)) for x, y in bucket],
                 fill=(234, 80, 50), outline=outline, width=max(1, round(scale * .008)))
    return np.asarray(image.resize((size, size), Image.Resampling.LANCZOS))


def generate_clip(spec: VideoSpec, seed: int) -> dict:
    """Return RGB frames and rendered bucket centroids in pixel coordinates, xy.

    A fixed excavator articulates two rigid links through a smooth lift/lower
    cycle. Each seed selects the initial phase and cycle speed. The camera,
    chassis, sand, and link lengths stay fixed. This kinematic scene supplies
    exact motion targets; it does not simulate soil or bucket contact forces.
    """
    if not isinstance(spec, VideoSpec):
        raise TypeError("spec must be a VideoSpec")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative int64")
    rng = np.random.default_rng(int(seed))
    initial_phase = float(rng.uniform(0, 2 * np.pi))
    angular_speed = float(rng.uniform(.8, 1.35))
    phase = initial_phase + angular_speed * np.arange(spec.frames) / spec.fps
    angles = np.stack((-.8 + .4 * np.sin(phase), 1.1 + .5 * np.sin(phase + .6)), axis=1)
    pivot = np.broadcast_to([.28, .60], (spec.frames, 2))
    elbow = pivot + .29 * np.stack((np.cos(angles[:, 0]), np.sin(angles[:, 0])), axis=1)
    tip = elbow + .30 * np.stack((np.cos(angles[:, 1]), np.sin(angles[:, 1])), axis=1)
    joints = np.stack((pivot, elbow, tip), axis=1)
    bucket_angles = .10 + .15 * np.sin(phase - .6)
    scene = _scene(spec.size)
    frames = np.stack([_render_frame(scene, points, angle, spec.size)
                       for points, angle in zip(joints, bucket_angles)])
    # Rasterization changes a polygon's pixel centroid slightly. Record the
    # exact displayed target using the same red weights as the evaluator.
    rgb = frames.astype(np.float64)
    weights = np.maximum(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) - 40, 0)
    rows, columns = np.mgrid[:spec.size, :spec.size]
    totals = weights.sum(axis=(1, 2))
    positions = (np.stack(((weights * columns).sum(axis=(1, 2)),
                           (weights * rows).sum(axis=(1, 2))), axis=1) / totals[:, None]).astype(np.float32)
    return {"frames": frames, "positions": positions,
            "trajectory_id": f"{GENERATOR_VERSION}-{int(seed):016x}",
            "joint_positions": (joints * spec.size).astype(np.float32),
            "initial_phase_radians": initial_phase,
            "angular_speed_radians_per_second": angular_speed}
