"""Flow matching and the visible Cosmos-Predict2.5 sampling loop.

The paper uses t=0 for data and t=1 for noise. The sampler below matches
Diffusers 0.39.0's Cosmos2_5_PredictBasePipeline, including its guidance
convention and the velocity of the observed prefix. UniPC supplies the
numerical solver; Euler is implemented separately for the chapter exercise.
"""
from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor


def make_noisy_latents(clean: Tensor, noise: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Return z_t=(1-t)z+t*noise and its velocity target noise-z.

    ``t`` may be scalar or broadcastable to the latent tensor. For one noise
    level per batch element, pass shape ``[B, 1, 1, 1, 1]``.
    """
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have the same shape")
    if not torch.isfinite(t).all() or torch.any((t < 0) | (t > 1)):
        raise ValueError("noise levels must be finite and in [0, 1]")
    return (1 - t) * clean + t * noise, noise - clean


def flow_matching_loss(prediction: Tensor, target: Tensor,
                       condition_mask: Tensor | None = None) -> Tensor:
    """Mean squared velocity error over generated elements only.

    A mask value of one identifies an observed element. Its gradient is zero.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    errors = (prediction.float() - target.float()).square()
    if condition_mask is None:
        return errors.mean()
    # Reduce the mask in the same FP32 precision as the squared errors. A
    # float16 count overflows once the future contains more than 65,504 values.
    mask = _expanded_mask(condition_mask, errors)
    weights = 1 - mask
    count = weights.sum()
    if count.item() == 0:
        raise ValueError("loss needs at least one generated element")
    return (errors * weights).sum() / count


def _expanded_mask(mask: Tensor, tensor: Tensor) -> Tensor:
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("condition_mask must contain only zero and one")
    return torch.broadcast_to(mask.to(device=tensor.device, dtype=tensor.dtype), tensor.shape)


def replace_conditioned(latents: Tensor, known: Tensor, mask: Tensor) -> Tensor:
    """Place observed latents in the Transformer input at every solver step."""
    if latents.shape != known.shape:
        raise ValueError("latents and known must have the same shape")
    expanded = _expanded_mask(mask, latents)
    return expanded * known + (1 - expanded) * latents


def euler_step(latents: Tensor, velocity: Tensor, t: float, t_next: float) -> Tensor:
    """Integrate dz/dt=v toward the next noise level (usually t_next < t)."""
    if latents.shape != velocity.shape:
        raise ValueError("latents and velocity must have the same shape")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (t, t_next)):
        raise ValueError("t and t_next must be finite and in [0, 1]")
    return latents + (t_next - t) * velocity


@torch.inference_mode()
def sample_cosmos_latents(
    transformer,
    scheduler,
    latents: Tensor,
    cond_latents: Tensor,
    condition_mask: Tensor,
    prompt_embeds: Tensor,
    negative_prompt_embeds: Tensor | None = None,
    *,
    num_steps: int = 15,
    guidance_scale: float = 7.0,
    padding_mask: Tensor | None = None,
    conditional_frame_timestep: float = 0.0001,
    callback: Callable[[int, float, Tensor], None] | None = None,
) -> Tensor:
    """Generate with the chapter Transformer and a supplied UniPC scheduler.

    At guidance > 1, Cosmos uses ``v_cond + g * (v_cond - v_negative)``.
    The known prefix is restored in every model input; its solver trajectory
    has velocity ``initial_noise - known``. This reproduces the reference
    loop instead of clamping the entire solver state after each step.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite")
    if latents.ndim != 5 or latents.shape != cond_latents.shape:
        raise ValueError("latents and cond_latents must have equal [B,C,T,H,W] shapes")
    if guidance_scale > 1 and negative_prompt_embeds is None:
        raise ValueError("guidance above one requires negative_prompt_embeds")
    _expanded_mask(condition_mask, latents)
    device, batch = latents.device, latents.shape[0]
    dtype = transformer.dtype
    mask = condition_mask.to(device=device, dtype=dtype)
    # This chapter conditions entire frames, so each frame has one noise level.
    indicator = condition_mask.to(device=device, dtype=torch.float32)[:, :, :, :1, :1]
    if not torch.equal(mask, indicator.expand_as(mask)):
        raise ValueError("Cosmos frame conditioning needs a spatially uniform mask per frame")
    scheduler.set_timesteps(num_steps, device=device)
    gt_velocity = (latents - cond_latents) * mask
    for index, timestep in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[index].expand(batch).to(device=device, dtype=torch.float32)
        model_t = (indicator * conditional_frame_timestep + (1 - indicator) * sigma.view(batch, 1, 1, 1, 1)
                   if conditional_frame_timestep >= 0 else sigma)
        model_input = (mask * cond_latents + (1 - mask) * latents).to(dtype)
        kwargs = dict(hidden_states=model_input, condition_mask=mask, timestep=model_t,
                      padding_mask=padding_mask, return_dict=False)
        velocity = transformer(encoder_hidden_states=prompt_embeds, **kwargs)[0]
        velocity = gt_velocity + velocity * (1 - mask)
        if guidance_scale > 1:
            negative = transformer(encoder_hidden_states=negative_prompt_embeds, **kwargs)[0]
            negative = gt_velocity + negative * (1 - mask)
            velocity = velocity + guidance_scale * (velocity - negative)
        latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
        if callback is not None:
            callback(index, float(timestep), latents)
    return latents
