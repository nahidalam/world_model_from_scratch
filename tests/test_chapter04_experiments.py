"""Prevent mixed or mislabeled artifacts from becoming a parity report."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from world_models.experiment import RunManifest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/chapter_04_experiments.py"
SPEC = spec_from_file_location("chapter04_experiments", SCRIPT)
experiments = module_from_spec(SPEC)
SPEC.loader.exec_module(experiments)


@pytest.fixture
def paired_runs(tmp_path, monkeypatch):
    args = SimpleNamespace(reference=tmp_path / "reference", scratch=tmp_path / "scratch",
                           output=tmp_path / "comparison")
    monkeypatch.setattr(experiments, "metadata", lambda: {"test_metadata": True})
    for implementation in ("reference", "scratch"):
        directory = getattr(args, implementation)
        directory.mkdir()
        manifest = RunManifest(
            checkpoint="nvidia/Cosmos-Predict2.5-2B", revision="test-revision",
            seed=0, num_frames=5, guidance_scale=7.0, num_inference_steps=1,
            height=16, width=16, prompt="a puck moves", observation="input.mp4",
            implementation=f"chapter04/{implementation}", implementation_sha256="a" * 64,
        )
        manifest.save(directory)
        (directory / "generation.json").write_text(json.dumps({
            "run_id": manifest.run_id, "implementation": implementation,
            "scheduler": {"solver_order": 2, "prediction_type": "flow_prediction"},
        }))
        np.save(directory / "frames.npy", np.zeros((5,16,16,3), dtype=np.uint8))
        torch.save(torch.zeros(1,16,2,2,2), directory / "latents_step_00.pt")
    return args


@pytest.mark.parametrize("artifact", ["rollout.mp4", "frames.npy", "manifest.json",
                                     "generation.json", "latents_step_07.pt"])
def test_generation_rejects_previous_artifacts_before_loading_gpu_models(tmp_path, artifact):
    (tmp_path / artifact).touch()
    with pytest.raises(ValueError, match="Choose a new output directory"):
        experiments.require_unused_generation_output(tmp_path)


def test_generation_allows_new_directories_and_existing_logs(tmp_path):
    experiments.require_unused_generation_output(tmp_path / "new-run")
    (tmp_path / "run.log").touch()
    experiments.require_unused_generation_output(tmp_path)


def test_compare_preserves_scheduler_and_source_report_identity(paired_runs):
    experiments.compare(paired_runs)
    report = json.loads((paired_runs.output / "comparison.json").read_text())
    assert report["decoded_uint8_max_abs"] == 0
    assert report["latent_comparison"]["latents_step_00.pt"]["max_abs"] == 0
    assert report["scheduler"]["solver_order"] == 2
    assert report["reference_generation_sha256"] == experiments.sha256(paired_runs.reference / "generation.json")


def test_compare_rejects_same_directory(paired_runs):
    paired_runs.scratch = paired_runs.reference
    with pytest.raises(ValueError, match="different directories"):
        experiments.compare(paired_runs)


def test_compare_rejects_swapped_implementation_labels(paired_runs):
    paired_runs.reference, paired_runs.scratch = paired_runs.scratch, paired_runs.reference
    with pytest.raises(ValueError, match="Expected a chapter04/reference"):
        experiments.compare(paired_runs)


@pytest.mark.parametrize("change,match", [
    ({"scheduler": {"solver_order": 1}}, "Scheduler configurations differ"),
    ({"scheduler": {}}, "needs a scheduler configuration"),
    ({"run_id": "old-run-id"}, "does not match its manifest"),
    ({"implementation": "reference"}, "does not match its manifest"),
])
def test_compare_rejects_incompatible_or_stale_generation_reports(paired_runs, change, match):
    path = paired_runs.scratch / "generation.json"
    record = json.loads(path.read_text())
    record.update(change)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=match):
        experiments.compare(paired_runs)


def test_compare_rejects_mixed_seed(paired_runs):
    path = paired_runs.scratch / "manifest.json"
    record = json.loads(path.read_text())
    record.pop("run_id")
    record["seed"] = 1
    RunManifest(**record).save(paired_runs.scratch)
    with pytest.raises(ValueError, match="configurations differ in seed"):
        experiments.compare(paired_runs)


def test_compare_rejects_extra_saved_steps(paired_runs):
    torch.save(torch.zeros(1,16,2,2,2), paired_runs.scratch / "latents_step_07.pt")
    with pytest.raises(ValueError, match="identical nonempty sets"):
        experiments.compare(paired_runs)
