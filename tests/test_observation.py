import numpy as np
import pytest

from world_models.observation import Observation


def test_rejects_float_frames() -> None:
    frames = np.zeros((4, 16, 16, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        Observation(frames)


def test_last_preserves_chronological_order() -> None:
    frames = np.arange(5, dtype=np.uint8).reshape(5, 1, 1, 1)
    frames = np.repeat(frames, 3, axis=-1)
    assert Observation(frames).last(2).frames[..., 0].ravel().tolist() == [3, 4]
