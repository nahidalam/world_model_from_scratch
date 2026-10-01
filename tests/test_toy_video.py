"""Verify the excavator mechanism, visible target, and physical clip timing."""
import numpy as np
import pytest

from world_models.toy_video import GENERATOR_VERSION, VideoSpec, generate_clip


@pytest.mark.parametrize("size", [32, 128])
def test_bucket_stays_visible_through_full_cycles(size):
    spec = VideoSpec(size=size, frames=129)
    clip = generate_clip(spec, 2**63 - 1)
    frames = clip["frames"].astype(np.float64)
    weights = np.maximum(frames[..., 0] - np.maximum(frames[..., 1], frames[..., 2]) - 40, 0)
    # Every frame contains a complete bucket, with a margin at image boundaries.
    assert np.all(weights.sum(axis=(1, 2)) > 0)
    assert not weights[:, (0, -1)].any()
    assert not weights[:, :, (0, -1)].any()
    yy, xx = np.mgrid[:size, :size]
    centers = np.stack(((weights * xx).sum(axis=(1, 2)),
                        (weights * yy).sum(axis=(1, 2))), axis=1) / weights.sum(axis=(1, 2))[:, None]
    np.testing.assert_allclose(centers, clip["positions"], atol=1e-5)
    assert clip["trajectory_id"].startswith(GENERATOR_VERSION)


def test_excavator_links_stay_connected_with_constant_lengths():
    size = 128
    clip = generate_clip(VideoSpec(size=size, frames=129), 17)
    joints = clip["joint_positions"]
    np.testing.assert_allclose(joints[:, 0], np.broadcast_to([.28 * size, .60 * size], (129, 2)))
    lengths = np.linalg.norm(np.diff(joints, axis=1), axis=-1)
    np.testing.assert_allclose(lengths, np.broadcast_to([.29 * size, .30 * size], (129, 2)), atol=1e-5)
    assert np.ptp(joints[:, 1, 0]) > 10
    assert np.ptp(joints[:, 2, 1]) > 25
    # Continuous kinematics also stay smooth where the arm reverses direction.
    assert np.linalg.norm(np.diff(joints[:, 2], axis=0), axis=-1).max() < 5


def test_frame_rate_preserves_motion_at_matching_times():
    slow = generate_clip(VideoSpec(frames=17, fps=8), 23)
    dense = generate_clip(VideoSpec(frames=33, fps=16), 23)
    np.testing.assert_array_equal(slow["frames"], dense["frames"][::2])
    np.testing.assert_array_equal(slow["positions"], dense["positions"][::2])


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5])
def test_generator_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="seed"):
        generate_clip(VideoSpec(), seed)
