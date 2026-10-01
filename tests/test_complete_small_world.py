"""Check that the one-file model computes exactly what the library computes."""
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from world_models import complete_small_world as one_file
from world_models.small_world import SmallWorldModel, sample_future, small_world_config
from world_models.training import _loss

ROOT = Path(__file__).resolve().parents[1]
TINY = one_file.Config(channels=3, frames=5, height=4, width=4, heads=2, head_dim=16,
                       blocks=2, context_dim=16, lora_rank=8, mlp_ratio=2.0)


def paired_models():
    """Build the library tiny preset and copy its weights into the one-file model."""
    torch.manual_seed(0)
    library = SmallWorldModel(small_world_config("tiny", (3, 5, 4, 4)))
    with torch.no_grad():
        library.transformer.proj_out.weight.normal_(std=0.02)
    model = one_file.WorldModel(TINY)
    copy_parameters(library, model)
    return library, model


def copy_parameters(source, target):
    pairs = list(zip(source.parameters(), target.parameters(), strict=True))
    with torch.no_grad():
        for a, b in pairs:
            assert a.shape == b.shape
            b.copy_(a)


@pytest.mark.parametrize("preset", ["small", "base"])
def test_presets_have_the_library_parameter_layout(preset):
    library = SmallWorldModel(small_world_config(preset))
    model = one_file.WorldModel(one_file.PRESETS[preset])
    assert [p.shape for p in library.parameters()] == [p.shape for p in model.parameters()]


def test_forward_matches_library():
    library, model = paired_models()
    noisy = torch.randn(2, 3, 5, 4, 4)
    mask = one_file.prefix_mask(noisy, 2)
    t = torch.tensor([0.2, 0.8])
    assert torch.equal(model(noisy, t, mask), library(noisy, t, mask))


def test_training_loss_and_gradients_match_library():
    library, model = paired_models()
    clean = torch.randn(2, 3, 5, 4, 4)
    ours = one_file.training_loss(model, clean, torch.Generator().manual_seed(5))
    theirs = _loss(library, clean, 2, torch.Generator().manual_seed(5), torch.device("cpu"), "fp32")
    assert torch.equal(ours, theirs)
    ours.backward()
    theirs.backward()
    for a, b in zip(library.parameters(), model.parameters(), strict=True):
        assert torch.equal(a.grad, b.grad)


def test_sampling_matches_library():
    library, model = paired_models()
    observed = torch.randn(1, 3, 2, 4, 4)
    ours = one_file.sample(model, observed, steps=4, seed=3)
    theirs = sample_future(library, observed, total_frames=5, steps=4, seed=3)
    assert torch.equal(ours, theirs)


def test_training_reduces_loss():
    torch.manual_seed(0)
    model = one_file.WorldModel(TINY)
    latents = torch.randn(4, 3, 5, 4, 4)
    losses = one_file.train(model, latents, steps=30, batch_size=2, accumulation=1, learning_rate=3e-3)
    assert losses[-1] < losses[0]


def test_chapter_listing_matches_source():
    chapter = (ROOT / "chapters/chapter_04/10-the-complete-model.md").read_text()
    listing = re.search(r"```python\n(.*?)```", chapter, re.S).group(1)
    assert listing == (ROOT / "src/world_models/complete_small_world.py").read_text()


SECTIONS = sorted((ROOT / "chapters/chapter_04").glob("0[1-9]-*.md"))


def definitions():
    """Yield (name, source) for each top-level definition the chapter builds."""
    import ast
    source = (ROOT / "src/world_models/complete_small_world.py").read_text()
    lines = source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign):
            name = node.targets[0].id
        if name is None or name == "main":
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        yield name, "".join(lines[start - 1:node.end_lineno]).rstrip("\n")


@pytest.mark.parametrize("name, source", list(definitions()), ids=lambda value: value if isinstance(value, str) and "\n" not in value else "")
def test_each_definition_is_built_in_exactly_one_section(name, source):
    homes = [path.name for path in SECTIONS if source in path.read_text()]
    assert len(homes) == 1, f"{name} appears verbatim in {homes or 'no section'}"


@pytest.mark.parametrize("path", SECTIONS, ids=lambda path: path.name)
def test_section_code_runs_and_matches_printed_output(path):
    import contextlib
    import io
    namespace = {}
    blocks = re.findall(r"```python\n(.*?)```(\n\n```text\n(.*?)```)?", path.read_text(), re.S)
    for code, _, expected in blocks:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, namespace)
        if expected:
            assert output.getvalue() == expected
