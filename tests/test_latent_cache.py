"""Exercise the dataset and complete cache lifecycle without downloading a VAE."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from world_models import latent_cache
from world_models.toy_video import VideoSpec, generate_clip, split_seeds


class FakeVAE:
    """Causal spatial/temporal subsampling with the WAN shape contract."""

    dtype = torch.float32

    def __init__(self, leaky=False):
        self.leaky = leaky
        self.config = SimpleNamespace(z_dim=16, latents_mean=np.linspace(-0.2, 0.2, 16).tolist(),
                                      latents_std=np.linspace(0.5, 1.5, 16).tolist())

    def encode(self, pixels):
        raw = pixels[:, :, ::4, ::8, ::8].repeat(1, 6, 1, 1, 1)[:, :16]
        if self.leaky:
            raw = raw + pixels.shape[2] * 0.2
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: raw))

    def decode(self, raw, return_dict=False):
        image = raw[:, :3].repeat_interleave(8, -1).repeat_interleave(8, -2)
        decoded = torch.cat((image[:, :, :1], image[:, :, 1:].repeat_interleave(4, 2)), dim=2)
        return (decoded,)


@pytest.fixture
def prepared_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(latent_cache, "_load_vae", lambda device: FakeVAE())
    path = tmp_path / "cache"
    spec = VideoSpec(size=32, frames=9, observed_frames=5)
    metadata = latent_cache.prepare_cache(path, "cpu", train_clips=3, val_clips=2,
                                          test_clips=2, spec=spec, seed=17)
    return path, metadata, spec


def test_generated_trajectories_are_deterministic_and_articulated():
    spec = VideoSpec(size=32, frames=65)
    first = generate_clip(spec, 11)
    again = generate_clip(spec, 11)
    other = generate_clip(spec, 12)
    assert first["frames"].shape == (65, 32, 32, 3)
    assert first["frames"].dtype == np.uint8
    assert first["positions"].dtype == np.float32
    np.testing.assert_array_equal(first["frames"], again["frames"])
    np.testing.assert_array_equal(first["positions"], again["positions"])
    assert first["trajectory_id"] == again["trajectory_id"] != other["trajectory_id"]
    assert not np.array_equal(first["positions"], other["positions"])
    movement = np.diff(first["positions"], axis=0)
    assert np.any(movement[:-1] * movement[1:] < 0)
    assert np.all(first["positions"] > 0)
    assert np.all(first["positions"] < spec.size - 1)
    # The supplied target is the rendered bucket, not the whole machine.
    frame = first["frames"][0].astype(float)
    weights = np.maximum(frame[..., 0] - np.maximum(frame[..., 1], frame[..., 2]) - 40, 0)
    y, x = np.mgrid[:spec.size, :spec.size]
    center = np.array([(weights * x).sum(), (weights * y).sum()]) / weights.sum()
    np.testing.assert_allclose(center, first["positions"][0], atol=0.1)


def test_splits_remain_disjoint_when_training_size_changes():
    train = split_seeds("train", 512, 12)
    val = split_seeds("val", 64, 12)
    test = split_seeds("test", 64, 12)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    np.testing.assert_array_equal(train[:16], split_seeds("train", 16, 12))
    with pytest.raises(ValueError, match="count"):
        split_seeds("train", 1_000_001)
    with pytest.raises(ValueError, match="seed"):
        split_seeds("train", 1, -1)


@pytest.mark.parametrize("kwargs", [{"size": 48.0}, {"size": 31}, {"frames": 16},
                                     {"observed_frames": 17}, {"observed_frames": 2}, {"fps": 0}])
def test_video_spec_rejects_unsupported_shapes(kwargs):
    with pytest.raises(ValueError):
        VideoSpec(**kwargs)


def test_cache_records_shapes_normalization_causality_and_artifacts(prepared_cache):
    path, metadata, spec = prepared_cache
    assert latent_cache.read_cache(path) == metadata
    assert metadata["latent_shape"] == [16, 3, 4, 4]
    assert metadata["observed_latent_frames"] == 2
    assert metadata["peak_allocated_bytes"] is None
    assert all(item["passed"] and item["max_abs"] == 0 for item in metadata["causality"])
    assert (path / "reconstruction.png").is_file()
    assert (path / "reconstruction.mp4").is_file()
    values = latent_cache.load_split(path, "train")
    assert values["latents"].shape == (3, 16, 3, 4, 4)
    assert values["seeds"].tolist() == [17, 18, 19]
    clip = generate_clip(spec, 17)
    np.testing.assert_array_equal(values["positions"][0].numpy(), clip["positions"])
    pixels = latent_cache._pixels(clip["frames"], torch.device("cpu"), torch.float32)
    raw = FakeVAE().encode(pixels).latent_dist.mode()
    mean, std = latent_cache._statistics(metadata["normalization"], torch.device("cpu"))
    torch.testing.assert_close(values["latents"][:1], (raw - mean) / std)


def test_decode_reverses_normalization_and_loads_only_one_vae(prepared_cache, monkeypatch):
    path, metadata, _ = prepared_cache
    calls = []
    def load(device):
        calls.append(device)
        return FakeVAE()
    monkeypatch.setattr(latent_cache, "_load_vae", load)
    latents = latent_cache.load_split(path, "val")["latents"]
    decoded = latent_cache.decode_latents(latents, metadata, "cpu")
    mean, std = latent_cache._statistics(metadata["normalization"], torch.device("cpu"))
    expected = latent_cache._uint8(FakeVAE().decode(latents * std + mean)[0])
    assert len(calls) == 1
    assert decoded.shape == (2, 9, 32, 32, 3)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, expected)


def test_modified_cache_is_rejected(prepared_cache):
    path, _, _ = prepared_cache
    with (path / "val.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch for val"):
        latent_cache.load_split(path, "val")


def test_preparation_preserves_existing_cache(prepared_cache, monkeypatch):
    path, metadata, spec = prepared_cache
    def unexpected_load(device):
        pytest.fail("existing cache must be checked before model loading")
    monkeypatch.setattr(latent_cache, "_load_vae", unexpected_load)
    with pytest.raises(ValueError, match="empty directory"):
        latent_cache.prepare_cache(path, "cpu", train_clips=1, val_clips=1, test_clips=1, spec=spec)
    assert latent_cache.read_cache(path) == metadata


def test_future_leakage_fails_before_publishing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(latent_cache, "_load_vae", lambda device: FakeVAE(leaky=True))
    path = tmp_path / "leaky"
    with pytest.raises(ValueError, match="causality check failed"):
        latent_cache.prepare_cache(path, "cpu", train_clips=1, val_clips=1, test_clips=1,
                                   spec=VideoSpec(size=32, frames=9))
    assert not (path / "cache.json").exists()
    with pytest.raises(FileNotFoundError):
        latent_cache.read_cache(path)


def test_cuda_failure_precedes_model_download_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "gpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        latent_cache.prepare_cache(path, "cuda", train_clips=1, val_clips=1, test_clips=1)
    assert not path.exists()


def test_cuda_selection_resolves_default_and_explicit_indices(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert latent_cache._device("cuda") == torch.device("cuda:2")
    assert latent_cache._device("cuda:1") == torch.device("cuda:1")
    assert selected == [torch.device("cuda:2"), torch.device("cuda:1")]
