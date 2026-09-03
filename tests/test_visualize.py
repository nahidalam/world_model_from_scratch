import numpy as np

from world_models.visualize import sample_frames, difference_map, disagreement_map


def test_sample_frames_hits_endpoints() -> None:
    frames = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    out = sample_frames(frames, 4)
    assert out.shape == (4, 4, 4, 3)


def test_identical_rollouts_have_zero_disagreement() -> None:
    a = np.full((3, 4, 4, 3), 128, dtype=np.uint8)
    assert disagreement_map([a, a.copy()]).max() == 0.0


def test_difference_map_is_bright_where_frames_differ() -> None:
    a = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    b = np.full((1, 2, 2, 3), 255, dtype=np.uint8)
    assert np.isclose(difference_map(a, b).mean(), 1.0)
