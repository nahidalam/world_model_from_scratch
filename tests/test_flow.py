"""Checks for the noise convention and observation/future boundary."""
import pytest

torch = pytest.importorskip("torch")

from world_models.flow import (euler_step, flow_matching_loss, make_noisy_latents,
                               replace_conditioned, sample_cosmos_latents)


def test_flow_endpoints_and_euler_sign():
    clean, noise = torch.tensor([2., -3.]), torch.tensor([1., 4.])
    for t, expected in [(0., clean), (1., noise)]:
        z_t, velocity = make_noisy_latents(clean, noise, torch.tensor(t))
        torch.testing.assert_close(z_t, expected)
        torch.testing.assert_close(euler_step(z_t, velocity, t, 0.), clean)


def test_observed_prefix_has_no_loss_or_gradient():
    prediction = torch.zeros(1, 2, 3, 2, 2, requires_grad=True)
    target = torch.ones_like(prediction)
    mask = torch.zeros(1, 1, 3, 2, 2)
    mask[:, :, :1] = 1
    loss = flow_matching_loss(prediction, target, mask)
    assert loss.item() == 1
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, :, :1]) == 0
    assert torch.count_nonzero(prediction.grad[:, :, 1:]) == 16
    with pytest.raises(ValueError, match="generated"):
        flow_matching_loss(prediction, target, torch.ones_like(mask))


def test_observation_replacement_preserves_future():
    z = torch.randn(1, 2, 3, 2, 2)
    known = torch.randn_like(z)
    mask = torch.zeros(1, 1, 3, 2, 2)
    mask[:, :, :1] = 1
    result = replace_conditioned(z, known, mask)
    torch.testing.assert_close(result[:, :, :1], known[:, :, :1])
    torch.testing.assert_close(result[:, :, 1:], z[:, :, 1:])


def test_half_precision_loss_counts_large_future_in_float32():
    prediction = torch.zeros(6, 16, 5, 16, 16, dtype=torch.float16, requires_grad=True)
    mask = torch.zeros(6, 1, 5, 16, 16, dtype=torch.float16)
    mask[:, :, :2] = 1
    loss = flow_matching_loss(prediction, torch.ones_like(prediction), mask)
    assert loss.dtype == torch.float32
    assert loss.item() == 1
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, :, :2]) == 0
    assert torch.all(prediction.grad[:, :, 2:] < 0)


def test_sampler_restores_history_and_uses_cosmos_guidance():
    class Model:
        dtype = torch.float32
        def __call__(self, hidden_states, condition_mask, encoder_hidden_states, **kwargs):
            assert torch.all(hidden_states[:, :, :1] == 2)
            return (torch.ones_like(hidden_states) * encoder_hidden_states.item(),)
    class Solver:
        def set_timesteps(self, n, device):
            self.timesteps = torch.tensor([1000.], device=device)
            self.sigmas = torch.tensor([1., 0.], device=device)
        def step(self, velocity, timestep, latents, return_dict):
            return (latents - velocity,)
    z = torch.zeros(1, 1, 2, 1, 1)
    known = torch.full_like(z, 2)
    mask = torch.tensor([1., 0.]).reshape(1, 1, 2, 1, 1)
    result = sample_cosmos_latents(Model(), Solver(), z, known, mask,
                                  torch.tensor(3.), torch.tensor(1.),
                                  num_steps=1, guidance_scale=7)
    # v=3+7*(3-1)=17 in the future; observed trajectory ends exactly at2.
    torch.testing.assert_close(result.flatten(), torch.tensor([2., -17.]))
