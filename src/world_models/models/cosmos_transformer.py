# Copyright 2025 The NVIDIA Team and The HuggingFace Team.
# Copyright 2026 World Models from Scratch contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0 .
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""A readable PyTorch implementation of the Cosmos-Predict2.5 Transformer.

Architecture and parameter names follow ``diffusers==0.39.0``:
https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/transformer_cosmos.py
The configuration defaults match nvidia/Cosmos-Predict2.5-2B on the
``diffusers/base/post-trained`` revision. No Diffusers code runs in this module.

We implement Predict2.5's text and video conditioning, including its unusual
output patch ordering. Transfer ControlNet and image-feature cross-attention
are outside this implementation and raise explicit errors if requested.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CosmosTransformerConfig:
    """The released 2B network configuration; smaller values support exercises."""

    in_channels: int = 17
    out_channels: int = 16
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    num_layers: int = 28
    mlp_ratio: float = 4.0
    text_embed_dim: int = 1024
    adaln_lora_dim: int = 256
    max_size: tuple[int, int, int] = (128, 240, 240)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    rope_scale: tuple[float, float, float] = (1.0, 3.0, 3.0)
    concat_padding_mask: bool = True
    extra_pos_embed_type: str | None = None
    use_crossattn_projection: bool = True
    crossattn_proj_in_channels: int = 100352
    encoder_hidden_states_channels: int = 1024
    controlnet_block_every_n: int | None = None
    img_context_dim_in: int | None = None
    img_context_num_tokens: int = 256
    img_context_dim_out: int = 2048

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    def __post_init__(self) -> None:
        if self.controlnet_block_every_n is not None or self.img_context_dim_in:
            raise ValueError("This chapter implements Predict2.5 without ControlNet or image-feature context.")
        if self.extra_pos_embed_type not in (None, "learnable"):
            raise ValueError("extra_pos_embed_type must be None or 'learnable'.")
        if self.attention_head_dim < 12 or self.attention_head_dim % 2:
            raise ValueError("3D RoPE requires an even attention_head_dim of at least 12.")
        for name in ("in_channels", "out_channels", "num_attention_heads", "num_layers", "adaln_lora_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        for name in ("max_size", "patch_size", "rope_scale"):
            values = tuple(getattr(self, name))
            if len(values) != 3 or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain three positive values.")
            object.__setattr__(self, name, values)
        if self.use_crossattn_projection and self.encoder_hidden_states_channels != self.text_embed_dim:
            raise ValueError("The projected text width must equal text_embed_dim.")


class RMSNorm(nn.Module):
    """Normalize each vector by its root mean square, then learn a scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


def patchify(x: Tensor, patch_size: tuple[int, int, int]) -> Tensor:
    """[B,C,T,H,W] -> [B,T/pt,H/ph,W/pw,C*pt*ph*pw]."""
    batch, channels, frames, height, width = x.shape
    pt, ph, pw = patch_size
    if frames % pt or height % ph or width % pw:
        raise ValueError(f"Video shape {tuple(x.shape)} is not divisible by patch_size={patch_size}.")
    x = x.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
    return x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)


def unpatchify_output(x: Tensor, grid: tuple[int, int, int], patch_size: tuple[int, int, int]) -> Tensor:
    """Decode output patches ordered [ph,pw,pt,C], as the checkpoint expects.

    The output projection learns this ordering. It differs from the [C,pt,ph,pw]
    ordering of input patches, so this function is not an inverse of patchify.
    """
    pt, ph, pw = patch_size
    x = x.unflatten(2, (ph, pw, pt, -1)).unflatten(1, grid)
    return x.permute(0, 7, 1, 6, 2, 4, 3, 5).flatten(6, 7).flatten(4, 5).flatten(2, 3)


class CosmosPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * math.prod(patch_size), out_channels, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(patchify(x, self.patch_size))


def timestep_embedding(timestep: Tensor, embedding_dim: int) -> Tensor:
    """Encode noise time with cosine/sine frequencies, separately from 3D RoPE."""
    if timestep.ndim != 1:
        raise ValueError("timestep_embedding expects a one-dimensional tensor.")
    half = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32)
    frequencies = torch.exp(exponent / half)
    angles = timestep[:, None].float() * frequencies[None, :]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(embedding, (0, embedding_dim % 2))


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(width, width, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(width, 3 * width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.activation(self.linear_1(x)))


class CosmosEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.t_embedder = CosmosTimestepEmbedding(width)
        self.norm = RMSNorm(width, eps=1e-6)

    def forward(self, hidden_states: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor]:
        projected = timestep_embedding(timestep, self.width).to(hidden_states)
        return self.t_embedder(projected), self.norm(projected)


class CosmosAdaLayerNorm(nn.Module):
    """Use a low-rank time projection to shift and scale a LayerNorm output."""

    def __init__(self, width: int, rank: int, *, gated: bool = False) -> None:
        super().__init__()
        self.width = width
        self.gated = gated
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()
        self.linear_1 = nn.Linear(width, rank, bias=False)
        self.linear_2 = nn.Linear(rank, (3 if gated else 2) * width, bias=False)

    def forward(self, x: Tensor, embedded_timestep: Tensor, temb: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        modulation = self.linear_2(self.linear_1(self.activation(embedded_timestep)))
        modulation = modulation + temb[..., :modulation.shape[-1]]
        if modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        parts = modulation.chunk(3 if self.gated else 2, dim=-1)
        shift, scale = parts[:2]
        normalized = self.norm(x) * (1 + scale) + shift
        return (normalized, parts[2]) if self.gated else normalized


class CosmosRotaryPosEmbed(nn.Module):
    """Allocate head dimensions to frame, row, and column coordinates."""

    def __init__(self, hidden_size: int, max_size: tuple[int, int, int],
                 patch_size: tuple[int, int, int], rope_scale: tuple[float, float, float],
                 base_fps: int = 24) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.max_size = tuple(size // patch for size, patch in zip(max_size, patch_size))
        self.base_fps = base_fps
        self.dim_h = self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w
        dims = (self.dim_t, self.dim_h, self.dim_w)
        self.theta = tuple(10000.0 * scale ** (dim / (dim - 2)) for dim, scale in zip(dims, rope_scale))

    def forward(self, x: Tensor, fps: int | None = None) -> tuple[Tensor, Tensor]:
        grid = tuple(size // patch for size, patch in zip(x.shape[2:], self.patch_size))
        if any(size > maximum for size, maximum in zip(grid, self.max_size)):
            raise ValueError(f"Patch grid {grid} exceeds RoPE maximum {self.max_size}.")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive.")
        seq = torch.arange(max(self.max_size), dtype=torch.float32, device=x.device)
        angles = []
        for axis, (size, dim, theta) in enumerate(zip(grid, (self.dim_t, self.dim_h, self.dim_w), self.theta)):
            positions = seq[:size]
            if axis == 0 and fps is not None:
                positions = positions / fps * self.base_fps
            frequency = 1.0 / theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
            axis_angles = torch.outer(positions, frequency)
            shape = [1, 1, 1, dim // 2]
            shape[axis] = size
            angles.append(axis_angles.reshape(shape).expand(*grid, dim // 2))
        phase = torch.cat(angles * 2, dim=-1).flatten(0, 2).float()
        return phase.cos(), phase.sin()


def apply_rotary_embedding(x: Tensor, rotary: tuple[Tensor, Tensor]) -> Tensor:
    """Rotate the two halves of each [B,heads,tokens,head_dim] vector."""
    cosine, sine = rotary
    real, imaginary = x.chunk(2, dim=-1)
    rotated = torch.cat((-imaginary, real), dim=-1)
    return (x.float() * cosine[None, None] + rotated.float() * sine[None, None]).to(x.dtype)


class CosmosAttention(nn.Module):
    """Project Q/K/V, normalize Q/K, optionally rotate, attend, and project out."""

    def __init__(self, width: int, heads: int, context_dim: int | None = None) -> None:
        super().__init__()
        self.heads = heads
        head_dim = width // heads
        context_dim = width if context_dim is None else context_dim
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(context_dim, width, bias=False)
        self.to_v = nn.Linear(context_dim, width, bias=False)
        self.norm_q = RMSNorm(head_dim, eps=1e-5)
        self.norm_k = RMSNorm(head_dim, eps=1e-5)
        self.to_out = nn.ModuleList([nn.Linear(width, width, bias=False), nn.Dropout(0.0)])

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor | None = None,
                attention_mask: Tensor | None = None,
                image_rotary_emb: tuple[Tensor, Tensor] | None = None) -> Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = self.to_q(hidden_states).unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = self.to_k(context).unflatten(2, (self.heads, -1)).transpose(1, 2)
        value = self.to_v(context).unflatten(2, (self.heads, -1)).transpose(1, 2)
        query, key = self.norm_q(query), self.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_embedding(query, image_rotary_emb)
            key = apply_rotary_embedding(key, image_rotary_emb)
        # PyTorch evaluates softmax(Q K^T / sqrt(head_dim)) V with a fused kernel.
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                                  dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).flatten(2, 3).to(query.dtype)
        return self.to_out[1](self.to_out[0](attended))


class GELUProjection(nn.Module):
    def __init__(self, width: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, inner, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.proj(x), approximate="none")


class CosmosFeedForward(nn.Module):
    def __init__(self, width: int, mlp_ratio: float) -> None:
        super().__init__()
        inner = int(width * mlp_ratio)
        self.net = nn.ModuleList([GELUProjection(width, inner), nn.Dropout(0.0), nn.Linear(inner, width, bias=False)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.net:
            x = layer(x)
        return x


class CosmosTransformerBlock(nn.Module):
    """Three gated residual branches: video attention, text attention, and MLP."""

    def __init__(self, config: CosmosTransformerConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.norm1 = CosmosAdaLayerNorm(width, config.adaln_lora_dim, gated=True)
        self.attn1 = CosmosAttention(width, config.num_attention_heads)
        self.norm2 = CosmosAdaLayerNorm(width, config.adaln_lora_dim, gated=True)
        self.attn2 = CosmosAttention(width, config.num_attention_heads, config.text_embed_dim)
        self.norm3 = CosmosAdaLayerNorm(width, config.adaln_lora_dim, gated=True)
        self.ff = CosmosFeedForward(width, config.mlp_ratio)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, embedded_timestep: Tensor,
                temb: Tensor, image_rotary_emb: tuple[Tensor, Tensor] | None = None,
                extra_pos_emb: Tensor | None = None, attention_mask: Tensor | None = None) -> Tensor:
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb
        normalized, gate = self.norm1(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.attn1(normalized, image_rotary_emb=image_rotary_emb)
        normalized, gate = self.norm2(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.attn2(normalized, encoder_hidden_states, attention_mask)
        normalized, gate = self.norm3(hidden_states, embedded_timestep, temb)
        return hidden_states + gate * self.ff(normalized)


class CosmosLearnablePositionalEmbed(nn.Module):
    """Optional additive positions for other Cosmos configurations, disabled in 2.5."""

    def __init__(self, config: CosmosTransformerConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        grid = [size // patch for size, patch in zip(config.max_size, config.patch_size)]
        self.pos_emb_t = nn.Parameter(torch.zeros(grid[0], config.hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(grid[1], config.hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(grid[2], config.hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        t, h, w = [size // patch for size, patch in zip(x.shape[2:], self.patch_size)]
        position = (self.pos_emb_t[:t][None, :, None, None] + self.pos_emb_h[:h][None, None, :, None]
                    + self.pos_emb_w[:w][None, None, None, :])
        position = position.expand(x.shape[0], -1, -1, -1, -1).flatten(1, 3)
        norm = torch.linalg.vector_norm(position, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=math.sqrt(norm.numel() / position.numel()))
        return (position / norm).to(x)


@dataclass
class CosmosTransformerOutput:
    sample: Tensor


class ScratchCosmosTransformer(nn.Module):
    """The full denoiser, compatible with released Diffusers state-dictionary keys.

    ``from_reference`` shares already-loaded weights through PyTorch's ``assign``
    option. Use it before attaching offload hooks, and discard the reference
    transformer afterward if independent storage is not required.
    """

    _no_split_modules = ["CosmosTransformerBlock"]

    def __init__(self, config: CosmosTransformerConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass either a config or configuration keywords.")
        self.config = config or CosmosTransformerConfig(**kwargs)
        self.gradient_checkpointing = False
        cfg = self.config
        width = cfg.hidden_size
        self.patch_embed = CosmosPatchEmbed(cfg.in_channels + int(cfg.concat_padding_mask), width, cfg.patch_size)
        self.rope = CosmosRotaryPosEmbed(cfg.attention_head_dim, cfg.max_size, cfg.patch_size, cfg.rope_scale)
        self.learnable_pos_embed = CosmosLearnablePositionalEmbed(cfg) if cfg.extra_pos_embed_type else None
        self.time_embed = CosmosEmbedding(width)
        self.transformer_blocks = nn.ModuleList([CosmosTransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm_out = CosmosAdaLayerNorm(width, cfg.adaln_lora_dim)
        self.proj_out = nn.Linear(width, math.prod(cfg.patch_size) * cfg.out_channels, bias=False)
        if cfg.use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(nn.Linear(cfg.crossattn_proj_in_channels, cfg.encoder_hidden_states_channels),
                                                nn.GELU())

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | CosmosTransformerConfig) -> ScratchCosmosTransformer:
        if isinstance(config, CosmosTransformerConfig):
            return cls(config)
        names = {field.name for field in fields(CosmosTransformerConfig)}
        unknown = [name for name in config if name not in names and not name.startswith("_")]
        if unknown:
            raise ValueError(f"Unsupported configuration fields: {unknown}")
        return cls(CosmosTransformerConfig(**{name: value for name, value in config.items() if name in names}))

    @classmethod
    def from_reference(cls, reference: nn.Module) -> ScratchCosmosTransformer:
        """Reuse released weights without allocating a second initialized 2B model."""
        with torch.device("meta"):
            model = cls.from_config(reference.config)
        model.load_state_dict(reference.state_dict(), strict=True, assign=True)
        model.train(reference.training)
        return model

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def forward(self, hidden_states: Tensor, timestep: Tensor, encoder_hidden_states: Tensor,
                block_controlnet_hidden_states: list[Tensor] | None = None,
                attention_mask: Tensor | None = None, fps: int | None = None,
                condition_mask: Tensor | None = None, padding_mask: Tensor | None = None,
                return_dict: bool = True) -> CosmosTransformerOutput | tuple[Tensor]:
        if block_controlnet_hidden_states is not None or not isinstance(encoder_hidden_states, Tensor):
            raise ValueError("Predict2.5 expects text features and no ControlNet residuals.")
        if hidden_states.ndim != 5:
            raise ValueError("hidden_states must have shape [B,C,T,H,W].")
        batch, _, frames, height, width = hidden_states.shape
        if condition_mask is not None:
            if condition_mask.shape != (batch, 1, frames, height, width):
                raise ValueError("condition_mask must have shape [B,1,T,H,W].")
            hidden_states = torch.cat((hidden_states, condition_mask.to(hidden_states)), dim=1)
        if hidden_states.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels} channels including condition_mask; got {hidden_states.shape[1]}.")
        if self.config.concat_padding_mask:
            if padding_mask is None or padding_mask.ndim != 4 or padding_mask.shape[:2] != (1, 1):
                raise ValueError("padding_mask must have shape [1,1,H,W], shared across the batch.")
            padding = F.interpolate(padding_mask.to(hidden_states), size=(height, width), mode="nearest")
            padding = padding.unsqueeze(2).expand(batch, 1, frames, height, width)
            hidden_states = torch.cat((hidden_states, padding), dim=1)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError("attention_mask must have shape [B,text_tokens].")
            if attention_mask.dtype != torch.bool and not attention_mask.is_floating_point():
                raise ValueError("Use a boolean attention mask or floating-point additive attention bias.")
            attention_mask = attention_mask[:, None, None, :]

        rotary = self.rope(hidden_states, fps)
        extra_position = self.learnable_pos_embed(hidden_states) if self.learnable_pos_embed is not None else None
        grid = tuple(size // patch for size, patch in zip((frames, height, width), self.config.patch_size))
        hidden_states = self.patch_embed(hidden_states).flatten(1, 3)
        if timestep.ndim == 1:
            if timestep.shape[0] not in (1, batch):
                raise ValueError("One-dimensional timestep must have one value or one per batch item.")
            temb, embedded = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5 and timestep.shape == (batch, 1, frames, 1, 1):
            if self.config.patch_size[0] != 1:
                raise ValueError("Per-frame timesteps require temporal patch size 1.")
            temb, embedded = self.time_embed(hidden_states, timestep.flatten())
            temb, embedded = (value.view(batch, grid[0], 1, 1, -1).expand(-1, -1, grid[1], grid[2], -1).flatten(1, 3)
                              for value in (temb, embedded))
        else:
            raise ValueError("timestep must have shape [B] or [B,1,T,1,1].")
        if self.config.use_crossattn_projection:
            encoder_hidden_states = self.crossattn_proj(encoder_hidden_states)
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                hidden_states = checkpoint(block, hidden_states, encoder_hidden_states, embedded,
                                           temb, rotary, extra_position, attention_mask,
                                           use_reentrant=False)
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, embedded, temb, rotary,
                                      extra_position, attention_mask)
        hidden_states = self.proj_out(self.norm_out(hidden_states, embedded, temb))
        sample = unpatchify_output(hidden_states, grid, self.config.patch_size)
        return CosmosTransformerOutput(sample) if return_dict else (sample,)
