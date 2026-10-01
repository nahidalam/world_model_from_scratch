import numpy as np
import pytest

from world_models.observation import Observation, load_video


def test_rejects_float_frames() -> None:
    frames = np.zeros((4, 16, 16, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        Observation(frames)


def test_last_preserves_chronological_order() -> None:
    frames = np.arange(5, dtype=np.uint8).reshape(5, 1, 1, 1)
    frames = np.repeat(frames, 3, axis=-1)
    assert Observation(frames).last(2).frames[..., 0].ravel().tolist() == [3, 4]


def test_video_loader_preserves_encoded_fps(tmp_path) -> None:
    import imageio.v3 as iio

    path = tmp_path / "clip.mp4"
    iio.imwrite(path, np.zeros((5, 16, 16, 3), dtype=np.uint8), fps=12)
    observation = load_video(path)
    assert observation.fps == 12
    assert observation.last(2).fps == 12
    assert observation.num_frames == 5
