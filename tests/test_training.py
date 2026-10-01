"""Check actual optimization, deterministic continuation, and run boundaries."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from world_models.small_world import SmallWorldModel, small_world_config
from world_models.training import (
    TrainConfig,
    load_training_checkpoint,
    resolve_device_precision,
    train_latents,
    validation_loss,
)


def cache():
    rng = torch.Generator().manual_seed(221)
    train = torch.randn(4, 2, 3, 4, 4, generator=rng)
    validation = torch.randn(3, 2, 3, 4, 4, generator=rng)
    metadata = {"latent_shape": [2, 3, 4, 4], "observed_latent_frames": 1, "cache_id": "test-cache-a"}
    return train, validation, metadata


def config(**kwargs):
    values = dict(preset="tiny", steps=6, batch_size=2, accumulation=2,
                  learning_rate=1e-3, evaluate_every=2, save_every=2,
                  precision="fp32", checkpoint_blocks=True)
    values.update(kwargs)
    return TrainConfig(**values)


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_identical(first, second)
    else:
        assert left == right


def test_interrupted_training_resumes_identical_model_optimizer_and_rng(tmp_path):
    train, validation, metadata = cache()
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    train_latents(train, validation, metadata, full, config())
    train_latents(train, validation, metadata, resumed, config(steps=3))
    report = train_latents(train, validation, metadata, resumed, config(), resume=resumed / "checkpoint.pt")
    first = load_training_checkpoint(full / "checkpoint.pt")
    second = load_training_checkpoint(resumed / "checkpoint.pt")
    for key in ("model", "optimizer", "scaler", "train_rng", "cpu_rng", "step", "skipped_updates"):
        assert_identical(first[key], second[key])
    assert report["start_step"] == 3
    assert report["end_step"] == 6
    assert report["effective_batch_size"] == 4


@pytest.mark.parametrize("mismatch", ["cache", "configuration"])
def test_resume_rejects_different_cache_or_training_configuration(tmp_path, mismatch):
    train, validation, metadata = cache()
    initial = config(steps=1)
    train_latents(train, validation, metadata, tmp_path, initial)
    resumed_config = replace(initial, steps=2)
    if mismatch == "cache":
        metadata = {**metadata, "cache_id": "test-cache-b"}
        expected = "same identity"
    else:
        resumed_config = replace(resumed_config, accumulation=3)
        expected = "configuration"
    with pytest.raises(ValueError, match=expected):
        train_latents(train, validation, metadata, tmp_path, resumed_config, resume=tmp_path / "checkpoint.pt")
    assert load_training_checkpoint(tmp_path / "checkpoint.pt")["step"] == 1


def test_new_run_preserves_existing_output(tmp_path):
    train, validation, metadata = cache()
    sentinel = tmp_path / "notes.txt"
    sentinel.write_text("existing result")
    with pytest.raises(FileExistsError, match="fresh output"):
        train_latents(train, validation, metadata, tmp_path, config(steps=1))
    assert sentinel.read_text() == "existing result"
    assert not (tmp_path / "checkpoint.pt").exists()


def test_fixed_validation_preserves_training_mode_and_global_rng():
    _, validation, metadata = cache()
    model = SmallWorldModel(small_world_config("tiny", tuple(metadata["latent_shape"]))).train()
    rng = torch.get_rng_state().clone()
    first = validation_loss(model, validation, 1, torch.device("cpu"), "fp32", batch_size=2)
    second = validation_loss(model, validation, 1, torch.device("cpu"), "fp32", batch_size=2)
    assert first == second
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training


def test_tiny_optimizer_run_reduces_fixed_probe_loss(tmp_path):
    # A short deterministic latent trajectory makes this a learning check,
    # independent of downloading the pretrained video compressor.
    train = torch.linspace(-1, 1, 3).reshape(1, 1, 3, 1, 1).expand(2, 2, 3, 4, 4).clone()
    metadata = {"latent_shape": [2, 3, 4, 4], "observed_latent_frames": 1, "cache_id": "learning-check"}
    report = train_latents(train, train.clone(), metadata, tmp_path,
                          config(steps=30, learning_rate=2e-3, accumulation=1,
                                 evaluate_every=30, save_every=30, overfit_clips=1,
                                 checkpoint_blocks=False))
    assert report["final_train_probe_loss"] < 0.8 * report["initial_train_probe_loss"]
    assert report["skipped_updates_total"] == 0


def test_bare_cuda_device_resolves_index_before_set_device(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    device, precision = resolve_device_precision("cuda", "fp32")
    assert selected == [torch.device("cuda:1")]
    assert device == torch.device("cuda:1")
    assert precision == "fp32"
