from world_models.experiment import RunManifest
import json
from pathlib import Path

import pytest

BASE = dict(
    checkpoint="nvidia/Cosmos-Predict2.5-2B", revision="diffusers/base/post-trained",
    seed=0, num_frames=29, guidance_scale=7.0, num_inference_steps=15,
    height=704, width=1280, prompt="a", observation="assets/chapter_02/sand_mining.mp4",
)


def test_same_config_same_run_id() -> None:
    assert RunManifest(**BASE).run_id == RunManifest(**BASE).run_id


def test_seed_changes_run_id() -> None:
    assert RunManifest(**BASE).run_id != RunManifest(**{**BASE, "seed": 1}).run_id


def test_run_id_ignores_environment() -> None:
    a = RunManifest(**BASE)
    b = RunManifest(**BASE, versions={"torch": "different"})
    assert a.run_id == b.run_id


def test_load_shipped_legacy_manifest_preserves_identifier() -> None:
    path = Path(__file__).resolve().parents[1] / "assets/chapter_02/seeds/rollout_seed0.manifest.json"
    manifest = RunManifest.load(path)
    assert manifest.run_id == "cbc346ed085d"
    assert manifest.conditioning_frames_sha256 is None


def test_manifest_round_trip_and_integrity_check(tmp_path) -> None:
    manifest = RunManifest(**BASE, conditioning_frames_sha256="a" * 64,
                           conditioning_num_frames=1, output_fps=16)
    path = manifest.save(tmp_path)
    assert RunManifest.load(path) == manifest
    record = json.loads(path.read_text())
    record["seed"] = 99
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="run_id mismatch"):
        RunManifest.load(path)


def test_new_conditioning_fields_change_identifier() -> None:
    original = RunManifest(**BASE)
    assert original.run_id == RunManifest(**BASE, conditioning_fps=None).run_id
    assert original.run_id != RunManifest(**BASE, conditioning_num_frames=1).run_id
    assert RunManifest(**BASE, conditioning_fps=8).run_id != RunManifest(
        **BASE, conditioning_fps=16).run_id


def test_implementation_identity_survives_round_trip_and_distinguishes_runs(tmp_path) -> None:
    """Reference and scratch rollouts must stay distinct under identical controls."""
    reference = RunManifest(**BASE, implementation="chapter04/reference",
                            implementation_sha256="a" * 64)
    scratch = RunManifest(**BASE, implementation="chapter04/scratch",
                          implementation_sha256="a" * 64)
    changed_code = RunManifest(**BASE, implementation="chapter04/scratch",
                               implementation_sha256="b" * 64)
    assert len({reference.run_id, scratch.run_id, changed_code.run_id}) == 3
    assert RunManifest.load(scratch.save(tmp_path)) == scratch
    # Unspecified implementation fields preserve identifiers from earlier chapters.
    assert RunManifest(**BASE).run_id == RunManifest(
        **BASE, implementation=None, implementation_sha256=None).run_id


@pytest.mark.parametrize("field,replacement", [
    ("implementation", "chapter04/reference"),
    ("implementation_sha256", "b" * 64),
])
def test_implementation_tampering_invalidates_saved_identifier(tmp_path, field, replacement) -> None:
    manifest = RunManifest(**BASE, implementation="chapter04/scratch",
                           implementation_sha256="a" * 64)
    path = manifest.save(tmp_path)
    record = json.loads(path.read_text())
    record[field] = replacement
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="run_id mismatch"):
        RunManifest.load(path)
