import numpy as np
from dataclasses import dataclass

from world_models.observation import Observation
from world_models.engine import RolloutEngine


@dataclass
class _FakeSpec:
    repo_id: str = "fake/model"
    revision: str = "main"
    height: int = 4
    width: int = 4
    fps: int = 16


class _FakeBackend:
    spec = _FakeSpec()

    def predict(self, observation, *, prompt, seed, num_frames,
                guidance_scale, num_inference_steps) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randint(0, 256, (num_frames, 4, 4, 3)).astype(np.uint8)


def test_engine_records_a_faithful_manifest() -> None:
    obs = Observation(np.zeros((5, 4, 4, 3), dtype=np.uint8))
    result = RolloutEngine(_FakeBackend()).generate(
        obs, prompt="x", seed=7, num_frames=9)
    assert result.frames.shape == (9, 4, 4, 3)
    assert result.manifest.seed == 7
    assert result.manifest.checkpoint == "fake/model"


def test_engine_is_reproducible_by_seed() -> None:
    obs = Observation(np.zeros((5, 4, 4, 3), dtype=np.uint8))
    engine = RolloutEngine(_FakeBackend())
    a = engine.generate(obs, prompt="x", seed=1).manifest.run_id
    b = engine.generate(obs, prompt="x", seed=1).manifest.run_id
    assert a == b
