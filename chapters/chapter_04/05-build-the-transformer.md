# 4.5 Build the Small Transformer

In the previous section, we defined what the model predicts: the velocity at
a noise level `t`. This section builds the Transformer model that makes the
prediction.

## The Model

The model predicts one step of generation. It takes three inputs: the noisy
latent, the noise level `t`, and a mask that marks which frames are
observed (Section 4.6 adds it). It returns a velocity with the latent's
shape. The sampler in Section 4.7 repeats this step to generate the future
frames that continue the observed video.

Inside the model, the inputs pass through three stages:

1. The patch layer from Section 4.2 turns the latent and mask into 320
   tokens.
2. The Transformer updates the tokens. It is a stack of identical layers
   called blocks: six in the small model and eight in the base model. Each
   block takes the 320 tokens and returns 320 updated tokens.
3. The output layer turns each token into 64 values, and ungrouping puts
   them back on the latent grid.

In code, the whole model is `SmallWorldModel`. We run the small model once
on a random latent and count its trainable parameters:

```python
import torch
from world_models.small_world import SmallWorldModel, small_world_config

model = SmallWorldModel(small_world_config("small"))
latents = torch.randn(1, 16, 5, 16, 16)
mask = torch.zeros(1, 1, 5, 16, 16)
mask[:, :, :2] = 1
velocity = model(latents, torch.tensor([0.5]), mask)
print(velocity.shape)
print(sum(parameter.numel() for parameter in model.parameters()))
```

```text
torch.Size([1, 16, 5, 16, 16])
7818240
```

The output has the latent's shape, so the model runs from input to output.
It has 7,818,240 trainable parameters, and 96% of them are in the blocks.
Next, we look inside one block.

## Inside a Block

![A Cosmos block applies adaptive normalization and gated residual additions around self-attention, context cross-attention, and a feed-forward network.](../../figures/chapter_04/transformer_block.svg)

*Figure 4.5: Tokens pass through three sublayers. The noise level sets each
sublayer's shift, scale, and gate, and the learned context enters through
cross-attention.*

Each block has three sublayers:

* Self-attention lets each token read the other tokens (Section 4.3).
* Cross-attention reads the context vector (Section 4.3).
* A feed-forward network transforms each token's features. In the base
  model, it expands 384 features to 1,536, applies GELU, and projects back
  to 384.

Each sublayer normalizes the tokens using `t`, computes its output, and adds
that output to the tokens through a gate. `CosmosTransformerBlock` in the
[book implementation](../../src/world_models/models/cosmos_transformer.py)
runs the three sublayers in order:

```python
def run_block(block, x, context, time_features, shared_time, rotary):
    normalized, gate = block.norm1(x, time_features, shared_time)
    x = x + gate * block.attn1(normalized, image_rotary_emb=rotary)

    normalized, gate = block.norm2(x, time_features, shared_time)
    x = x + gate * block.attn2(normalized, context)

    normalized, gate = block.norm3(x, time_features, shared_time)
    return x + gate * block.ff(normalized)
```

`rotary` holds the position rotations from Section 4.3. `time_features` and
`shared_time` carry the noise level. The next section shows how the block
uses them.

## How a Block Uses the Noise Level

The model's input mixes the clean latent with noise. At `t = 0.1`, the input
is mostly the clean latent. At `t = 0.9`, it is mostly noise. The model must
know how much noise its input holds to predict the velocity. So every
sublayer receives `t`.

In `run_block`, each sublayer starts with a line like
`normalized, gate = block.norm1(x, time_features, shared_time)`. This
section explains that line in two steps. First, we turn `t` into features.
Second, the features set a scale, a shift, and a gate for the sublayer.

First, the features. The noise level is one number. `timestep_embedding`
turns it into sine and cosine values at several frequencies. Each frequency
responds to `t` at its own rate, so together they give the network a vector
that describes `t`. Learned layers then turn this vector into
`time_features` and `shared_time`.

Second, adaptive layer normalization. Layer normalization rescales the
token's features. Then a scale and a shift computed from `t` adjust them:

$$
\widetilde{x}=\operatorname{LayerNorm}(x)(1+s(t))+b(t).
$$

The gate `g(t)` sets how much of the sublayer's output joins the token:

$$
x_{\text{next}}=x+g(t)\odot f(\widetilde{x}).
$$

The example below runs both steps on six tokens, using the features of
`t = 0.9`:

```python
import torch
from torch import nn
from world_models.models.cosmos_transformer import timestep_embedding

torch.manual_seed(0)
x = torch.randn(1, 6, 24)
time_features = timestep_embedding(torch.tensor([0.9]), 24)
modulation = nn.Sequential(
    nn.SiLU(),
    nn.Linear(24, 8, bias=False),
    nn.Linear(8, 3 * 24, bias=False),
)
shift, scale, gate = modulation(time_features).chunk(3, dim=-1)
normalized = nn.functional.layer_norm(x, (24,), eps=1e-6)
normalized = normalized * (1 + scale[:, None]) + shift[:, None]

sublayer = nn.Linear(24, 24, bias=False)
y = x + gate[:, None] * sublayer(normalized)
print(y.shape)
```

```text
torch.Size([1, 6, 24])
```

In the model, `norm1`, `norm2`, and `norm3` each compute their own scale,
shift, and gate this way.

Next, we give the model the observed frames it must continue to generate a
video.
