from world_models.experiment import RunManifest

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
