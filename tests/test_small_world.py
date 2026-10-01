"""Verify the trainable video model's gradients, memory options, and sampler."""
from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from world_models.small_world import (
    SmallWorldModel,
    prefix_mask,
    sample_future,
    small_world_config,
)


def tiny_model():
    torch.manual_seed(17)
    model = SmallWorldModel(small_world_config("tiny", (3, 5, 4, 4)))
    # A nonzero output projection exposes every branch to the first backward.
    with torch.no_grad():
        model.transformer.proj_out.weight.normal_(std=0.02)
    return model


def inputs():
    noisy = torch.randn(2, 3, 5, 4, 4)
    return noisy, torch.tensor([0.2, 0.8]), prefix_mask(noisy, 2)


def test_future_loss_reaches_observations_and_learned_context():
    model = tiny_model()
    noisy, times, mask = inputs()
    noisy.requires_grad_()
    prediction = model(noisy, times, mask)
    assert prediction.shape == noisy.shape
    prediction[:, :, 2:].square().mean().backward()
    assert torch.isfinite(noisy.grad).all()
    assert noisy.grad[:, :, :2].abs().sum() > 0
    assert torch.isfinite(model.null_context.grad).all()
    assert model.null_context.grad.abs().sum() > 0


def test_checkpointing_preserves_forward_and_parameter_gradients():
    regular = tiny_model()
    checkpointed = deepcopy(regular)
    checkpointed.transformer.gradient_checkpointing = True
    noisy, times, mask = inputs()
    expected = regular(noisy, times, mask)
    actual = checkpointed(noisy, times, mask)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for plain, saved in zip(regular.parameters(), checkpointed.parameters()):
        assert plain.grad is not None and saved.grad is not None
        torch.testing.assert_close(saved.grad, plain.grad)


def test_bfloat16_autocast_with_float32_parameters_and_checkpointing():
    model = tiny_model()
    model.transformer.gradient_checkpointing = True
    noisy, times, mask = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = model(noisy, times, mask)
        loss = prediction.float().square().mean()
    assert prediction.dtype == torch.bfloat16
    loss.backward()
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("training", [False, True])
def test_sampler_preserves_prefix_mode_and_rng_and_repeats_seed(training):
    model = tiny_model().train(training)
    observed = torch.randn(1, 3, 2, 4, 4)
    seen = []
    hook = model.register_forward_pre_hook(lambda module, args: seen.append(args[0][:, :, :2].clone()))
    rng_before = torch.random.get_rng_state().clone()
    first = sample_future(model, observed, total_frames=5, steps=3, seed=72)
    second = sample_future(model, observed, total_frames=5, steps=3, seed=72)
    different = sample_future(model, observed, total_frames=5, steps=3, seed=73)
    hook.remove()
    assert model.training == training
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first[:, :, 2:], different[:, :, 2:])
    torch.testing.assert_close(first[:, :, :2], observed, rtol=0, atol=0)
    assert len(seen) == 9
    for prefix in seen:
        torch.testing.assert_close(prefix, observed, rtol=0, atol=0)


def test_sampler_restores_training_mode_after_failed_forward():
    model = tiny_model().train()
    with pytest.raises(ValueError, match="precision"):
        sample_future(model, torch.zeros(1, 3, 2, 4, 4), total_frames=5, precision="invalid")
    assert model.training


def test_prefix_and_preset_errors_explain_valid_shapes():
    with pytest.raises(ValueError, match="preset"):
        small_world_config("other")
    with pytest.raises(ValueError, match="even"):
        small_world_config("tiny", (3, 5, 5, 4))
    with pytest.raises(ValueError, match="future"):
        prefix_mask(torch.zeros(1, 3, 5, 4, 4), 5)
    with pytest.raises(ValueError, match="shorter prefix"):
        sample_future(tiny_model(), torch.zeros(1, 3, 5, 4, 4), total_frames=5)
