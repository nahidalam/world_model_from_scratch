"""Check motion metrics and the complete CPU artifact lifecycle with a fake VAE."""
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from world_models import latent_cache
from world_models.experiment import RunManifest
from world_models.small_world_evaluation import TRACKING, evaluate_checkpoint, trajectory_metrics
from world_models.toy_video import VideoSpec, generate_clip
from world_models.training import TrainConfig, file_sha256, load_training_checkpoint, train_latents


@pytest.mark.parametrize("size", [32, 128])
def test_ground_truth_tracks_real_trajectory_better_than_persistence(size):
    spec = VideoSpec(size=size)
    clip = generate_clip(spec, 17)
    truth = clip["frames"]
    persistence = truth.copy()
    persistence[spec.observed_frames:] = truth[spec.observed_frames - 1]
    exact = trajectory_metrics(truth, truth, clip["positions"], spec.observed_frames)
    baseline = trajectory_metrics(persistence, truth, clip["positions"], spec.observed_frames)
    assert exact["future_pixel_rmse"] == 0
    assert exact["object_detection_fraction"] == 1
    assert exact["centroid_error_penalized_px"] < 0.1
    assert baseline["object_detection_fraction"] == 1
    assert baseline["future_pixel_rmse"] > 0
    assert baseline["centroid_error_penalized_px"] > 0.03 * size
    assert exact["future_frames"] == spec.frames - spec.observed_frames
    assert exact["final_centroid_error_penalized_px"] == exact["per_frame_penalized_error_px"][-1]
    assert exact["final_frame_detected"] is True


def test_missing_foreground_receives_diagonal_penalty():
    spec = VideoSpec(size=32, frames=9)
    clip = generate_clip(spec, 19)
    missing = np.full_like(clip["frames"], [16, 20, 28])
    result = trajectory_metrics(missing, clip["frames"], clip["positions"], spec.observed_frames)
    assert result["object_detection_fraction"] == 0
    assert result["centroid_error_detected_px"] is None
    assert result["centroid_error_penalized_px"] == pytest.approx(math.hypot(spec.size, spec.size))
    assert result["estimated_centers"] == [None] * 4
    assert result["per_frame_detected"] == [False] * 4
    assert result["future_pixel_rmse"] > 0
    assert result["final_centroid_error_penalized_px"] == pytest.approx(math.hypot(spec.size, spec.size))
    assert result["final_frame_detected"] is False


def test_bucket_tracker_ignores_separate_small_red_regions():
    spec = VideoSpec()
    clip = generate_clip(spec, 17)
    truth = clip["frames"]
    predicted = truth.copy()
    # The sky is separate from the bucket. A small red artifact there should
    # leave the center of the largest connected bucket silhouette unchanged.
    predicted[spec.observed_frames:, 2:4, 2:4] = [234, 80, 50]
    exact = trajectory_metrics(truth, truth, clip["positions"], spec.observed_frames)
    noisy = trajectory_metrics(predicted, truth, clip["positions"], spec.observed_frames)
    assert noisy["object_detection_fraction"] == 1
    assert noisy["estimated_centers"] == exact["estimated_centers"]
    assert noisy["future_pixel_rmse"] > exact["future_pixel_rmse"]


@pytest.mark.parametrize("region", ["one_pixel", "whole_frame"])
def test_bucket_detection_rejects_regions_outside_reference_area(region):
    spec = VideoSpec()
    clip = generate_clip(spec, 17)
    predicted = np.zeros_like(clip["frames"])
    if region == "one_pixel":
        predicted[:, 64, 64] = [234, 80, 50]
    else:
        predicted[:] = [234, 80, 50]
    result = trajectory_metrics(predicted, clip["frames"], clip["positions"], spec.observed_frames)
    assert result["object_detection_fraction"] == 0
    assert result["centroid_error_penalized_px"] == pytest.approx(math.hypot(spec.size, spec.size))


def test_observed_pixels_and_positions_do_not_enter_future_metrics():
    spec = VideoSpec(size=32, frames=9)
    clip = generate_clip(spec, 23)
    exact = trajectory_metrics(clip["frames"], clip["frames"], clip["positions"], spec.observed_frames)
    changed = clip["frames"].copy()
    changed[:spec.observed_frames] = 255
    positions = clip["positions"].copy()
    positions[:spec.observed_frames] = -100
    observed_changes = trajectory_metrics(changed, clip["frames"], positions, spec.observed_frames)
    assert exact == observed_changes


def test_generator_source_mismatch_stops_before_loading_or_regenerating(tmp_path, monkeypatch):
    from world_models import toy_video

    metadata = {"source_sha256": {"toy_video.py": "0" * 64}}
    monkeypatch.setattr(latent_cache, "read_cache", lambda path: metadata)

    def unexpected(*args, **kwargs):
        pytest.fail("source identity must be checked before loading latents or regenerating targets")

    monkeypatch.setattr(latent_cache, "load_split", unexpected)
    monkeypatch.setattr(toy_video, "generate_clip", unexpected)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="generator source recorded in the cache"):
        evaluate_checkpoint(tmp_path / "cache", tmp_path / "checkpoint.pt", output,
                            device="cpu", examples=1)
    assert not output.exists()


class LifecycleVAE:
    """Small causal codec used to exercise orchestration and real media writing."""

    dtype = torch.float32
    config = SimpleNamespace(z_dim=16, latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

    def encode(self, pixels):
        raw = pixels[:, :, ::4, ::8, ::8].repeat(1, 6, 1, 1, 1)[:, :16]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: raw))

    def decode(self, raw, return_dict=False):
        images = raw[:, :3].repeat_interleave(8, -1).repeat_interleave(8, -2)
        return (torch.cat((images[:, :, :1], images[:, :, 1:].repeat_interleave(4, 2)), dim=2),)


def test_cache_train_sample_manifest_and_real_media_roundtrip(tmp_path, monkeypatch):
    loads = []

    def load_vae(device):
        loads.append(device)
        return LifecycleVAE()

    monkeypatch.setattr(latent_cache, "_load_vae", load_vae)
    data, training, output = (tmp_path / name for name in ("data", "training", "evaluation"))
    spec = VideoSpec(size=32, frames=9, observed_frames=5)
    metadata = latent_cache.prepare_cache(data, "cpu", train_clips=2, val_clips=1,
                                         test_clips=2, spec=spec, seed=31)
    train = latent_cache.load_split(data, "train")["latents"]
    validation = latent_cache.load_split(data, "val")["latents"]
    train_latents(train, validation, metadata, training,
                  TrainConfig(preset="tiny", steps=2, batch_size=1, accumulation=1,
                              precision="fp32", evaluate_every=2, save_every=2))
    checkpoint = training / "checkpoint.pt"
    report = evaluate_checkpoint(data, checkpoint, output, device="cpu", examples=2,
                                 sampling_steps=2, seed=111, precision="fp32")
    assert len(loads) == 2  # One cache encoder, then one evaluation decoder.
    assert report["cache_id"] == metadata["cache_id"]
    assert report["checkpoint_sha256"] == file_sha256(checkpoint)
    assert report["observed_latent_max_error"] == 0
    assert report["split"] == "test"
    assert report["generator"] == metadata["generator"]
    assert report["tracking"] == TRACKING
    assert report["sampling"]["peak_allocated_bytes"] is None
    assert report["decoding"]["peak_allocated_bytes"] is None
    assert json.loads((output / "evaluation.json").read_text()) == report
    held_out = latent_cache.load_split(data, "test")
    for index, seed in enumerate(held_out["seeds"].tolist()):
        directory = output / f"trajectory_{seed}"
        manifest = RunManifest.load(directory / "manifest.json")
        assert Path(manifest.checkpoint) == checkpoint.resolve()
        assert manifest.revision == report["checkpoint_sha256"]
        assert manifest.run_id == report["trajectories"][index]["run_id"]
        inference = json.loads((directory / "inference.json").read_text())
        assert inference == report["inference"]
        assert inference["sampling_precision"] == "fp32"
        assert inference["decoder_dtype"] == "torch.float32"
        assert inference["vae_source"] == metadata["source"]
        assert inference["cache_id"] == metadata["cache_id"]
        assert inference["source"]["latent_cache.py"] == file_sha256(Path(latent_cache.__file__))
        assert inference["source"]["toy_video.py"] == metadata["source_sha256"]["toy_video.py"]
        digest = hashlib.sha256(json.dumps(inference, sort_keys=True).encode()).hexdigest()
        assert manifest.implementation_sha256 == digest
        for field, value in (("sampling_precision", "bf16"), ("decoder_dtype", "torch.bfloat16")):
            changed_inference = {**inference, field: value}
            changed_digest = hashlib.sha256(json.dumps(changed_inference, sort_keys=True).encode()).hexdigest()
            assert replace(manifest, implementation_sha256=changed_digest).run_id != manifest.run_id
        assert manifest.conditioning_num_frames == 5
        assert manifest.num_frames == 9
        assert manifest.seed == 111 + index
        truth = generate_clip(spec, seed)["frames"]
        assert manifest.conditioning_frames_sha256 == hashlib.sha256(truth[:5].tobytes()).hexdigest()
        for filename in manifest.outputs:
            assert (directory / filename).is_file()
        for filename in ("rollout.mp4", "ground_truth.mp4", "persistence.mp4"):
            video = iio.imread(directory / filename)
            assert video.shape == (9, 32, 32, 3)
            assert video.dtype == np.uint8
        with Image.open(directory / "comparison.png") as image:
            assert image.size == (128, 168)
            image.verify()
        frames = np.load(directory / "frames.npy", allow_pickle=False)
        assert frames.shape == (9, 32, 32, 3)
        sampled = torch.load(directory / "latents.pt", weights_only=True, map_location="cpu")
        torch.testing.assert_close(sampled[:, :2], held_out["latents"][index, :, :2], rtol=0, atol=0)
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics == report["trajectories"][index]
        assert metrics["generated"]["future_frames"] == 4
    # The manifest loader checks content identity, including the checkpoint hash.
    path = output / f"trajectory_{held_out['seeds'][0].item()}" / "manifest.json"
    changed = json.loads(path.read_text())
    changed["revision"] = "different-checkpoint"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="run_id mismatch"):
        RunManifest.load(path)
    # Evaluation validates cache/checkpoint pairing before creating result files.
    wrong = load_training_checkpoint(checkpoint)
    wrong["cache_id"] = "different-cache"
    wrong_path = tmp_path / "wrong-checkpoint.pt"
    torch.save(wrong, wrong_path)
    with pytest.raises(ValueError, match="cache used for training"):
        evaluate_checkpoint(data, wrong_path, tmp_path / "wrong-evaluation", device="cpu", examples=1)
    assert not (tmp_path / "wrong-evaluation").exists()
