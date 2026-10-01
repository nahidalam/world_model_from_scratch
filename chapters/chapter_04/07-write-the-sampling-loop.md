# 4.7 Write the Sampling Loop

The model from Section 4.5 looks at a noisy latent and predicts its
velocity. The velocity tells us how to remove a little noise: moving the
latent against it brings the latent closer to a clean video. One prediction
moves the latent one small step. To generate the three future latent frames, we start them as
noise and take 30 steps toward a clean latent.

Each step does three things:

1. Put the two observed latent frames back in place, using the mask from
   Section 4.6.
2. Ask the model for the velocity at the current noise level `t`.
3. Move the latent a small distance against the velocity, and lower `t` by
   1/30.

The noise level starts at `t = 1`, pure noise, and reaches `t = 0` after 30
steps. `sample` runs these three steps, numbered in the comments:

```python
import torch
from torch import Tensor

from world_models.complete_small_world import PRESETS, WorldModel, prefix_mask


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


model = WorldModel(PRESETS["small"])
observed = torch.randn(1, 16, 2, 16, 16)  # two observed latent frames
future = sample(model, observed)
print(future.shape)  # same shape as the latent
print(torch.equal(future[:, :, :2], observed))  # observed frames unchanged
```

```text
torch.Size([1, 16, 5, 16, 16])
True
```

The model is still untrained, so the three future latent frames are noise.

Next, we train the model so these steps produce real motion.
