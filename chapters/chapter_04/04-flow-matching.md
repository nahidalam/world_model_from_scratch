# 4.4 Implement Flow Matching

In the previous section, attention let each token read the other tokens.
Now we decide what the model outputs.

One option is to predict the future latent in one step. In real video, the
same five frames can lead to many futures. A model trained this way learns
their average, and the average of several arm positions is a blur.

Flow matching takes a different approach. It starts from noise and trains
the model to move the noisy latent a small step toward the clean latent.
The model repeats this step until the latent is clean. Cosmos uses this
method, and so do we.

This section defines what the model predicts and how we score it. Section
4.8 uses these definitions to train the model.

![Training mixes a clean latent with noise and compares the predicted velocity with noise minus the clean latent; generation starts from noise and repeatedly steps toward t = 0.](../../figures/chapter_04/flow_matching.svg)

*Figure 4.4: Top, training: mix a clean latent (blue) with noise (orange) at
level t, and compare the predicted velocity with ε − z. Bottom, generation:
start from noise and take small steps toward t = 0 until a clean latent
remains.*

## Model Input and Target

For each cached latent `z`, we draw standard normal noise `ε` and a noise
level `t` between zero and one. The model's input mixes the two:

$$
z_t=(1-t)z+t\epsilon.
$$

At `t = 0` the input is the clean latent. At `t = 1` it is pure noise. The
target is the direction from the clean latent to the noise, called the
velocity:

$$
v=\epsilon-z.
$$

The sign follows the
[Cosmos flow-matching formulation](https://arxiv.org/html/2511.00062v1#S3.SS1),
and the sampler in Section 4.7 uses the same convention.
`make_noisy_latents` computes the input and target for a batch shaped like
our cache, with one noise level per clip:

```python
import torch
from world_models.flow import make_noisy_latents

torch.manual_seed(0)
clean = torch.randn(2, 16, 5, 16, 16)  # two cached latents
noise = torch.randn_like(clean)
t = torch.rand(2).reshape(2, 1, 1, 1, 1)  # one noise level per clip

noisy, target = make_noisy_latents(clean, noise, t)
print(noisy.shape, target.shape)
print(torch.allclose(noisy - t * target, clean, atol=1e-6))
```

```text
torch.Size([2, 16, 5, 16, 16]) torch.Size([2, 16, 5, 16, 16])
True
```

The last line steps back from the input to the clean latent with the exact
velocity, which confirms the sign. During generation, the model predicts
the velocity, and the sampler takes small steps with it.

## Loss

The loss compares the model's velocity with the target using mean squared
error. `flow_matching_loss` computes it:

```python
from world_models.flow import flow_matching_loss

print(flow_matching_loss(target, target).item())
print(flow_matching_loss(target + 1, target).item())
```

```text
0.0
1.0
```

A prediction equal to the target gives zero. An error of one at every value
gives one. Section 4.6 adds a mask so the loss counts only the frames to
generate.

## Noise Levels

Generation starts at `t = 1` and ends at `t = 0`, so the model must predict
the velocity at every level. We draw each clip's noise level uniformly
between zero and one, as `torch.rand` does above. The
[Cosmos scheduler](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/schedulers/rectified_flow.py)
can also shift the levels toward high noise. Our model uses uniform levels.

Next, we build the Transformer that makes the prediction.
