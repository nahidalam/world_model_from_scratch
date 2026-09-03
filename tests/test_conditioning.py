import numpy as np
from PIL import Image

from world_models.observation import Observation
from world_models.backends.cosmos import to_conditioning


def test_conditioning_returns_one_pil_frame_per_input() -> None:
    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    conditioning = to_conditioning(Observation(frames))

    assert len(conditioning) == 8
    assert all(isinstance(f, Image.Image) for f in conditioning)
    assert conditioning[0].size == (64, 64)  # unmodified; the pipeline resizes
