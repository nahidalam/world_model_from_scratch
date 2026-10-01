# 4.7 Write the Sampling Loop

The model predicts one small step: the velocity at one noise level (Section
4.5). To generate the future frames, we start from noise at `t = 1` and
repeat the step until `t = 0`. Each step moves the latent by the velocity
times the gap between noise levels:

$$
z_{t_{\text{next}}}=z_t+(t_{\text{next}}-t)v_\theta(z_t,t,c).
$$

The noise level falls at each step, so `t_next - t` is negative and the
step moves toward the clean latent. Before each step, the loop restores the
observed frames (Section 4.6). The loop runs 30 steps:

```python
import torch
from world_models.flow import euler_step
from world_models.small_world import SmallWorldModel, prefix_mask, small_world_config

model = SmallWorldModel(small_world_config("small"))
observed = torch.randn(1, 16, 2, 16, 16)  # two observed latent frames
latent = torch.randn(1, 16, 5, 16, 16)  # start from noise
known = torch.zeros_like(latent)
known[:, :, :2] = observed
mask = prefix_mask(latent, observed_frames=2)
levels = torch.linspace(1, 0, 31)  # 30 steps from t = 1 to t = 0

with torch.no_grad():
    for t, t_next in zip(levels[:-1], levels[1:]):
        latent = mask * known + (1 - mask) * latent
        velocity = model(latent, t.expand(1), mask)
        latent = euler_step(latent, velocity, t, t_next)
latent = mask * known + (1 - mask) * latent

print(latent.shape)
print(torch.equal(latent[:, :, :2], observed))
```

```text
torch.Size([1, 16, 5, 16, 16])
True
```

The result has the latent's shape, and the observed frames stay as given.
The model here is untrained, so the three future frames are still noise.
Section 4.8 trains it.

[`sample_future`](../../src/world_models/small_world.py) runs the same loop
and adds seeded noise, device placement, and mixed precision, so results can
be reproduced. Decoding its output with the VAE gives 17 RGB frames
(Section 4.9). More steps follow the velocity more closely and take longer.
We use 30 steps and save the step count and seed with each result.

Next, we train the model.
