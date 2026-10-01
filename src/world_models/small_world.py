"""An observation-conditioned video Transformer for the Chapter 4 training lab."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn

from world_models.flow import euler_step, replace_conditioned
from world_models.models.cosmos_transformer import CosmosTransformerConfig, ScratchCosmosTransformer


def small_world_config(preset: str = "base", latent_shape: tuple[int, ...] = (16, 5, 16, 16)) -> CosmosTransformerConfig:
    """Size the same architecture for a cached [C,T,H,W] latent volume."""
    presets = {"tiny": (2, 16, 2, 16, 8, 2.0),
               "small": (4, 64, 6, 256, 64, 4.0),
               "base": (6, 64, 8, 256, 64, 4.0)}
    if preset not in presets:
        raise ValueError(f"Choose a preset from {tuple(presets)}.")
    if len(latent_shape) != 4 or any(n < 1 for n in latent_shape):
        raise ValueError("latent_shape must contain positive C,T,H,W dimensions.")
    channels, frames, height, width = latent_shape
    if frames < 2 or height % 2 or width % 2:
        raise ValueError("Use at least two latent frames and even spatial dimensions.")
    heads, head_dim, layers, context, rank, ratio = presets[preset]
    return CosmosTransformerConfig(
        in_channels=channels + 1, out_channels=channels, num_attention_heads=heads,
        attention_head_dim=head_dim, num_layers=layers, mlp_ratio=ratio,
        text_embed_dim=context, adaln_lora_dim=rank, max_size=(frames, height, width),
        patch_size=(1, 2, 2), rope_scale=(1.0, 1.0, 1.0), concat_padding_mask=False,
        use_crossattn_projection=False, encoder_hidden_states_channels=context,
    )


class SmallWorldModel(nn.Module):
    """Train the denoiser from random weights with one learned context token.

    The context token supplies a shared learned bias. Observed video frames
    provide the information that distinguishes one trajectory from another.
    """

    def __init__(self, config: CosmosTransformerConfig) -> None:
        super().__init__()
        if config.concat_padding_mask or config.use_crossattn_projection or config.patch_size[0] != 1:
            raise ValueError("SmallWorldModel uses direct context, temporal patches of one, and a uniform video grid.")
        self.transformer = ScratchCosmosTransformer(config)
        self.null_context = nn.Parameter(torch.randn(1, 1, config.text_embed_dim) * 0.02)
        # Starting with zero velocity gives the first update a controlled scale.
        nn.init.zeros_(self.transformer.proj_out.weight)

    @property
    def config(self) -> CosmosTransformerConfig:
        return self.transformer.config

    def forward(self, noisy: Tensor, t: Tensor, condition_mask: Tensor) -> Tensor:
        batch, _, frames, _, _ = noisy.shape
        if t.shape != (batch,):
            raise ValueError("Supply one noise level per example with shape [B].")
        indicator = condition_mask[:, :, :, :1, :1].float()
        if not torch.equal(condition_mask, indicator.expand_as(condition_mask)):
            raise ValueError("The conditioning mask must identify whole frames.")
        times = indicator * 0.0001 + (1 - indicator) * t.reshape(batch, 1, 1, 1, 1)
        return self.transformer(
            hidden_states=noisy, timestep=times, condition_mask=condition_mask,
            encoder_hidden_states=self.null_context.expand(batch, -1, -1), return_dict=False,
        )[0]


def prefix_mask(latents: Tensor, observed_frames: int) -> Tensor:
    if latents.ndim != 5 or not 0 < observed_frames < latents.shape[2]:
        raise ValueError("Choose a nonempty observed prefix followed by at least one future frame.")
    mask = latents.new_zeros((latents.shape[0], 1, *latents.shape[2:]))
    mask[:, :, :observed_frames] = 1
    return mask


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision not in ("bf16", "fp16"):
        raise ValueError("precision must be fp32, bf16, or fp16.")
    if device.type != "cuda":
        raise ValueError("The chapter mixed-precision profiles use CUDA. Choose fp32 on CPU.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device supports the fp16 profile; choose --precision fp16.")
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


@torch.inference_mode()
def sample_future(model: SmallWorldModel, observed: Tensor, *, total_frames: int,
                  steps: int = 30, seed: int = 0, precision: str = "fp32") -> Tensor:
    """Integrate velocity from noise to video, receiving only the observed prefix."""
    if observed.ndim != 5 or not 0 < observed.shape[2] < total_frames:
        raise ValueError("observed must be [B,C,T_observed,H,W] with a shorter prefix than total_frames.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    device = next(model.parameters()).device
    observed = observed.to(device=device, dtype=torch.float32)
    shape = (observed.shape[0], observed.shape[1], total_frames, *observed.shape[3:])
    rng = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(shape, generator=rng).to(device)
    known = torch.zeros_like(latents)
    known[:, :, :observed.shape[2]] = observed
    mask = prefix_mask(latents, observed.shape[2])
    was_training = model.training
    model.eval()
    try:
        for index in range(steps):
            t, next_t = 1 - index / steps, 1 - (index + 1) / steps
            model_input = replace_conditioned(latents, known, mask)
            with autocast_context(device, precision):
                velocity = model(model_input, torch.full((shape[0],), t, device=device), mask)
            latents = euler_step(latents, velocity.float(), t, next_t)
        return replace_conditioned(latents, known, mask)
    finally:
        model.train(was_training)
