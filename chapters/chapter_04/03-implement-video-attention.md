# 4.3 Implement Video Attention

In the previous section, we created tokens from video input. Each token
holds the information of one patch. The bucket's token needs the arm's pose
and the bucket's earlier positions to predict its next position. Attention
lets each token read information from the other tokens.

Open the [architecture explorer](../../interactive/chapter_04/model_explorer.html)
and choose an attention query. Follow its weights to see which tokens
contribute to its output.

![Queries and keys determine attention weights; those weights combine values from tokens throughout the video grid.](../../figures/chapter_04/attention.svg)

*Figure 4.3: Each token is projected into a query, key, and value. Queries
and keys, rotated by position, give the attention weights, and the weights
combine the values into the output.*

## One Attention Head

We project each token into a query, a key, and a value. The query describes
what the token looks for. The key describes what the token offers. The
value holds the information the token passes on. We compare each query with
every key to get a weight per token. The weights then combine the values.

For queries `Q`, keys `K`, and values `V`, one head calculates:

$$
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right),
\qquad Y = AV.
$$

Here `d` is the number of features in one head. Two tokens let us inspect
every weight:

```python
import math
import torch

q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
k = q.clone()
v = torch.tensor([[10.0, 0.0], [0.0, 20.0]])

scores = q @ k.T / math.sqrt(q.shape[-1])
weights = scores.softmax(dim=-1)
output = weights @ v
print(weights.round(decimals=3))
print(output.round(decimals=3))
```

```text
tensor([[0.6700, 0.3300],
        [0.3300, 0.6700]])
tensor([[ 6.6980,  6.6050],
        [ 3.3020, 13.3950]])
```

The first query equals the first key, so their score is the highest.
Softmax turns the scores into weights of 0.67 and 0.33. Each row sums to
one, so each output is a weighted average of the values. The first output
is 0.67 × [10, 0] + 0.33 × [0, 20], or about [6.7, 6.6]. This is
[scaled dot-product attention](https://arxiv.org/abs/1706.03762).

In our model, each query compares with every key in the clip, including
keys from later frames. The model refines all future frames at once, so
each future frame uses the current estimate of the others.

## Multi-Head Attention

A single head gives each token one set of weights. The bucket's token needs
information from two places: the arm's patches and the bucket's earlier
frames. Multi-head attention runs several heads at once, each with its own
weights. One head can follow the arm while another follows the bucket. The
base model splits each token's 384 features into six heads of 64 features,
runs attention in each head, and joins the results back into 384 features.

## Token Positions

Attention compares token content. The model also needs each token's
position: its frame, row, and column. Before comparing, the model prepares
queries and keys in two steps.

First, RMS normalization divides each query and key by its root mean
square. This keeps their values in the same range.

Second, rotary position embeddings, or RoPE, rotate pairs of query and key
features by an angle that depends on the token's position. As in Cosmos,
each head's features split into time, height, and width rotations. The dot
product of two rotated vectors then depends on the offset between their
positions, as the [RoPE derivation](https://arxiv.org/abs/2104.09864) shows.

## Cross-Attention

Cross-attention lets video tokens read a second input, the context. Queries
come from the video tokens. Keys and values come from the context. In
Cosmos, the context is a text prompt.

Our model uses the observed frames in place of a prompt. They show the
scene and how the arm moves. So the context is one vector that we train
with the other weights, and every example uses the same vector. With one
context token, each query gives it a weight of one. The cross-attention
output is therefore the same for every token. We keep the layer so our
block matches Cosmos. The same code then loads the released weights in
Section 4.A.

Next, we decide what the model should predict with that information.
