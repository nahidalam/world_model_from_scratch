"""Chapter 4's small world model in one file.

This file collects everything Sections 4.2 to 4.8 build, in reading order:
tokens, attention, flow matching, the Transformer, conditioning, sampling,
and training. The library modules world_models.small_world and
world_models.models.cosmos_transformer compute the same outputs; they add
input checks, mixed precision, and the options needed to load the released
Cosmos checkpoint. tests/test_complete_small_world.py confirms the match.

Train on a prepared latent cache (Section 4.8 creates it):

    python -m world_models.complete_small_world \
        --data outputs/chapter_04/excavator/data --preset small --steps 200
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# 4.1 Choose the model and the world -------------------------------------------

@dataclass(frozen=True)
class Config:
    channels: int = 16        # VAE latent channels
    frames: int = 5           # latent frames per clip
    height: int = 16          # latent height
    width: int = 16           # latent width
    observed_frames: int = 2  # latent frames encoding the five observed video frames
    heads: int = 6
    head_dim: int = 64
    blocks: int = 8
    context_dim: int = 256
    lora_rank: int = 64
    mlp_ratio: float = 4.0

    @property
    def hidden(self) -> int:
        return self.heads * self.head_dim


PRESETS = {"small": Config(heads=4, blocks=6), "base": Config()}


# 4.2 Turn video into latent tokens --------------------------------------------

def patchify(x: Tensor) -> Tensor:
    """[B, C, T, H, W] -> [B, T * H/2 * W/2, 4C], one channel's 2 x 2 values at a time."""
    b, c, t, h, w = x.shape
    x = x.reshape(b, c, t, h // 2, 2, w // 2, 2)
    return x.permute(0, 2, 3, 5, 1, 4, 6).reshape(b, t * (h // 2) * (w // 2), 4 * c)


def unpatchify(x: Tensor, frames: int, height: int, width: int) -> Tensor:
    """[B, tokens, 4C] ordered (row, column, channel) -> [B, C, T, H, W]."""
    x = x.reshape(x.shape[0], frames, height // 2, width // 2, 2, 2, -1)
    return x.permute(0, 6, 1, 2, 4, 3, 5).reshape(x.shape[0], -1, frames, height, width)


# 4.3 Implement video attention ------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps) * self.weight


def rotary_angles(cfg: Config, device: torch.device) -> tuple[Tensor, Tensor]:
    """Cosine and sine of each token's rotation, [tokens, head_dim]."""
    dim_hw = cfg.head_dim // 6 * 2
    dims = (cfg.head_dim - 2 * dim_hw, dim_hw, dim_hw)  # time, row, column
    grid = (cfg.frames, cfg.height // 2, cfg.width // 2)
    angles = []
    for axis, (size, dim) in enumerate(zip(grid, dims)):
        frequency = 1.0 / 10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim)
        axis_angles = torch.outer(torch.arange(size, device=device).float(), frequency)
        shape = [1, 1, 1, dim // 2]
        shape[axis] = size
        angles.append(axis_angles.reshape(shape).expand(*grid, dim // 2))
    phase = torch.cat(angles * 2, dim=-1).flatten(0, 2)
    return phase.cos(), phase.sin()


def rotate(x: Tensor, rotary: tuple[Tensor, Tensor]) -> Tensor:
    """Rotate pairs of features in [B, heads, tokens, head_dim] by position."""
    cosine, sine = rotary
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return (x.float() * cosine + rotated.float() * sine).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, context_dim: int | None = None) -> None:
        super().__init__()
        self.heads = heads
        context_dim = context_dim or width
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(context_dim, width, bias=False)
        self.to_v = nn.Linear(context_dim, width, bias=False)
        self.norm_q = RMSNorm(width // heads, eps=1e-5)
        self.norm_k = RMSNorm(width // heads, eps=1e-5)
        self.to_out = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, context: Tensor | None = None,
                rotary: tuple[Tensor, Tensor] | None = None) -> Tensor:
        context = x if context is None else context
        q = self.to_q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = self.to_k(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v = self.to_v(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q, k = self.norm_q(q), self.norm_k(k)
        if rotary is not None:
            q, k = rotate(q, rotary), rotate(k, rotary)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.to_out(y.transpose(1, 2).flatten(2).to(q.dtype))


# 4.5 Build the small Transformer ----------------------------------------------

def timestep_features(t: Tensor, width: int) -> Tensor:
    """Sine and cosine of the noise level at `width / 2` frequencies."""
    half = width // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    angles = t[:, None].float() * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class TimeEmbedding(nn.Module):
    """Return per-sublayer time features and a shared time projection."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(width, width, bias=False)
        self.linear_2 = nn.Linear(width, 3 * width, bias=False)
        self.norm = RMSNorm(width, eps=1e-6)

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        features = timestep_features(t, self.linear_1.in_features)
        return self.norm(features), self.linear_2(F.silu(self.linear_1(features)))


class AdaptiveNorm(nn.Module):
    """LayerNorm with a shift, scale, and (optionally) gate set by the noise level."""

    def __init__(self, width: int, rank: int, parts: int = 3) -> None:
        super().__init__()
        self.parts = parts
        self.linear_1 = nn.Linear(width, rank, bias=False)  # low rank: AdaLN-LoRA
        self.linear_2 = nn.Linear(rank, parts * width, bias=False)

    def forward(self, x: Tensor, time_features: Tensor, shared_time: Tensor) -> tuple[Tensor, Tensor | None]:
        modulation = self.linear_2(self.linear_1(F.silu(time_features)))
        modulation = modulation + shared_time[..., :modulation.shape[-1]]
        shift, scale, *gate = modulation.chunk(self.parts, dim=-1)
        normalized = F.layer_norm(x, x.shape[-1:], eps=1e-6) * (1 + scale) + shift
        return normalized, (gate[0] if gate else None)


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        width, inner = cfg.hidden, int(cfg.hidden * cfg.mlp_ratio)
        self.norm1 = AdaptiveNorm(width, cfg.lora_rank)
        self.self_attention = Attention(width, cfg.heads)
        self.norm2 = AdaptiveNorm(width, cfg.lora_rank)
        self.cross_attention = Attention(width, cfg.heads, cfg.context_dim)
        self.norm3 = AdaptiveNorm(width, cfg.lora_rank)
        self.feed_forward = nn.Sequential(nn.Linear(width, inner, bias=False), nn.GELU(),
                                          nn.Linear(inner, width, bias=False))

    def forward(self, x: Tensor, context: Tensor, time_features: Tensor, shared_time: Tensor,
                rotary: tuple[Tensor, Tensor]) -> Tensor:
        normalized, gate = self.norm1(x, time_features, shared_time)
        x = x + gate * self.self_attention(normalized, rotary=rotary)
        normalized, gate = self.norm2(x, time_features, shared_time)
        x = x + gate * self.cross_attention(normalized, context)
        normalized, gate = self.norm3(x, time_features, shared_time)
        return x + gate * self.feed_forward(normalized)


class WorldModel(nn.Module):
    """Predict the velocity of every latent value from a noisy latent and its mask."""

    def __init__(self, cfg: Config = PRESETS["base"]) -> None:
        super().__init__()
        self.cfg = cfg
        width = cfg.hidden
        self.patch_embed = nn.Linear(4 * (cfg.channels + 1), width, bias=False)
        self.time_embed = TimeEmbedding(width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.blocks))
        self.norm_out = AdaptiveNorm(width, cfg.lora_rank, parts=2)
        self.proj_out = nn.Linear(width, 4 * cfg.channels, bias=False)
        nn.init.zeros_(self.proj_out.weight)  # start from zero velocity
        self.context = nn.Parameter(torch.randn(1, 1, cfg.context_dim) * 0.02)

    def forward(self, noisy: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        b, _, frames, height, width = noisy.shape
        # 4.6: observed frames are clean, so they get a near-zero noise level.
        observed = mask[:, 0, :, 0, 0]
        frame_t = observed * 0.0001 + (1 - observed) * t[:, None]
        x = self.patch_embed(patchify(torch.cat((noisy, mask), dim=1)))
        tokens_per_frame = (height // 2) * (width // 2)
        time_features, shared_time = (
            y.view(b, frames, 1, -1).expand(-1, -1, tokens_per_frame, -1).flatten(1, 2)
            for y in self.time_embed(frame_t.flatten()))
        rotary = rotary_angles(self.cfg, noisy.device)
        context = self.context.expand(b, -1, -1)
        for block in self.blocks:
            x = block(x, context, time_features, shared_time, rotary)
        x, _ = self.norm_out(x, time_features, shared_time)
        return unpatchify(self.proj_out(x), frames, height, width)


# 4.6 Condition on observations (with the flow-matching target from 4.4) ------

def prefix_mask(latents: Tensor, observed_frames: int) -> Tensor:
    """1 for observed frames, 0 for frames to generate, shaped [B, 1, T, H, W]."""
    mask = latents.new_zeros((latents.shape[0], 1, *latents.shape[2:]))
    mask[:, :, :observed_frames] = 1
    return mask


def training_loss(model: WorldModel, clean: Tensor, generator: torch.Generator | None = None) -> Tensor:
    noise = torch.randn(clean.shape, generator=generator).to(clean.device)
    t = torch.rand(clean.shape[0], generator=generator).to(clean.device)
    mask = prefix_mask(clean, model.cfg.observed_frames)
    level = t[:, None, None, None, None]
    noisy = (1 - level) * clean + level * noise
    model_input = mask * clean + (1 - mask) * noisy
    prediction = model(model_input, t, mask)
    future = (1 - mask).expand_as(clean)
    return ((prediction - (noise - clean)).square() * future).sum() / future.sum()


# 4.7 Write the sampling loop --------------------------------------------------

@torch.no_grad()
def sample(model: WorldModel, observed: Tensor, steps: int = 30, seed: int = 0) -> Tensor:
    """Generate the future latent frames that follow `observed` [B, C, T_obs, H, W]."""
    b, c, _, height, width = observed.shape
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn((b, c, model.cfg.frames, height, width), generator=generator).to(observed.device)
    known = torch.zeros_like(latent)
    known[:, :, :observed.shape[2]] = observed
    mask = prefix_mask(latent, observed.shape[2])
    for i in range(steps):
        t, t_next = 1 - i / steps, 1 - (i + 1) / steps
        latent = mask * known + (1 - mask) * latent  # 1. restore observed frames
        velocity = model(latent, torch.full((b,), t, device=latent.device), mask)  # 2. predict
        latent = latent + (t_next - t) * velocity  # 3. take one step
    return mask * known + (1 - mask) * latent


# 4.8 Train a small world model ------------------------------------------------

def train(model: WorldModel, latents: Tensor, *, steps: int, batch_size: int = 4,
          accumulation: int = 8, learning_rate: float = 3e-4, seed: int = 17) -> list[float]:
    """Run `steps` optimizer updates of `batch_size * accumulation` clips each."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False)
    generator = torch.Generator().manual_seed(seed + 1)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            batch = latents[torch.randint(len(latents), (batch_size,), generator=generator)]
            loss = training_loss(model, batch.to(device), generator)
            (loss / accumulation).backward()
            total += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(total / accumulation)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="Latent cache directory from Section 4.8")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    latents = torch.load(args.data / "train.pt", map_location="cpu", weights_only=True)["latents"]
    torch.manual_seed(17)
    model = WorldModel(PRESETS[args.preset]).to(args.device)
    if tuple(latents.shape[1:]) != (model.cfg.channels, model.cfg.frames, model.cfg.height, model.cfg.width):
        raise ValueError(f"Cached latents have shape {tuple(latents.shape[1:])}; expected [16, 5, 16, 16].")
    losses = train(model, latents, steps=args.steps)
    print(f"loss: first update {losses[0]:.4f}, last update {losses[-1]:.4f}")
    observed = latents[:1, :, :model.cfg.observed_frames].to(args.device)
    print("generated latent:", tuple(sample(model, observed).shape))


if __name__ == "__main__":
    main()
