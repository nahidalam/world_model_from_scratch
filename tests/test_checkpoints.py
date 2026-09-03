from world_models.backends.cosmos import CHECKPOINTS


def test_specs_declare_a_positive_context_length() -> None:
    assert all(spec.num_context_frames >= 1 for spec in CHECKPOINTS.values())


def test_revisions_are_pinned() -> None:
    assert all(spec.revision for spec in CHECKPOINTS.values())
