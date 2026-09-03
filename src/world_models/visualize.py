"""Model-agnostic rollout inspection. Operates on (T, H, W, 3) uint8 frames."""
from __future__ import annotations

import numpy as np


def sample_frames(frames: np.ndarray, k: int) -> np.ndarray:
    """Return k roughly evenly spaced frames from a (T, H, W, C) stack."""
    if frames.ndim != 4:
        raise ValueError(f"expected (T, H, W, C), got shape {frames.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    t = frames.shape[0]
    idx = np.linspace(0, t - 1, min(k, t)).round().astype(int)
    return frames[idx]


def difference_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute per-pixel difference between two equal-shape stacks.

    Returns (T, H, W) float in [0, 1], bright where the two disagree. Use it
    two ways: between two rollouts to localize divergence, or between a stack
    and its own shifted copy to localize motion.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0
    return diff.mean(axis=-1)


def disagreement_map(futures: list[np.ndarray]) -> np.ndarray:
    """Per-pixel standard deviation across sampled futures.

    Each future is (T, H, W, C) uint8 of identical shape. Returns (T, H, W)
    float, bright where the sampled pixels differ. This is not a calibrated
    uncertainty or confidence estimate.
    """
    if len(futures) < 2:
        raise ValueError("need at least two futures to measure disagreement")
    shapes = {f.shape for f in futures}
    if len(shapes) != 1:
        raise ValueError(f"futures must share one shape, got {shapes}")
    stack = np.stack(futures).astype(np.float32) / 255.0   # (N, T, H, W, C)
    return stack.std(axis=0).mean(axis=-1)                 # (N collapsed) -> (T, H, W)
