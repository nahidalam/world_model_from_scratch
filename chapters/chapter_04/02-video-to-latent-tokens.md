# 4.2 Turn Video into Latent Tokens

A Transformer reads a sequence of vectors called tokens. Our input video
clip contains 278,528 pixel positions. Because attention compares every
token with every other token, using one token per pixel would be far too
expensive.

Instead, a frozen VAE compresses the video into a much smaller latent
representation. We then group nearby latent cells into patches, producing
320 tokens that the Transformer can process.

![The VAE encoder compresses 17 frames into a 16-channel latent of 5 × 16 × 16; grouping 1 × 2 × 2 cells gives 320 patches, and the input projection turns each into a 384-feature token.](../../figures/chapter_04/latent_tokens.svg)

*Figure 4.2: The VAE compresses the video in time and space. Grouping
neighboring latent cells gives 320 patches, and the input projection turns
each patch into a token.*

## Compress the Video with the VAE

Nearby pixels and frames look alike. The VAE removes this repetition by
compressing the video frames.
[WAN2.1's causal VAE](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py)
keeps the first frame. It merges each following group of four frames into
one latent frame. It also shrinks height and width by eight. We write video
tensors as `[batch, channels, frames, height, width]`:

```python
frames, height, width = 17, 128, 128
latent_frames = (frames - 1) // 4 + 1
latent_height = height // 8
latent_width = width // 8
tokens = latent_frames * (latent_height // 2) * (latent_width // 2)

print((16, latent_frames, latent_height, latent_width))
print(tokens)
```

```text
(16, 5, 16, 16)
320
```

The five observed frames become the first two latent frames, and the model
generates the other three.

## Group Latent Cells into Patches

The latent has 5 × 16 × 16 = 1,280 cells. Each cell could be a token, but
we group them to cut the count further. We will make a patch of 2 × 2
neighboring cells in one frame, so we get 320 tokens. Attention compares
every pair of tokens, so four times fewer tokens means sixteen times fewer
comparisons.

The model learns which value sits where in a patch, so the grouping order
must stay fixed. A 4 × 4 grid numbered 0 to 15 shows where each value lands:

```python
import torch

x = torch.arange(16).reshape(1, 1, 1, 4, 4)
b, c, f, h, w = x.shape

patches = x.reshape(b, c, f, h // 2, 2, w // 2, 2)
patches = patches.permute(0, 2, 3, 5, 1, 4, 6)
patches = patches.reshape(b, f * (h // 2) * (w // 2), 4 * c)
print(patches[0])
```

```text
tensor([[ 0,  1,  4,  5],
        [ 2,  3,  6,  7],
        [ 8,  9, 12, 13],
        [10, 11, 14, 15]])
```

Each row is one patch. The first row holds 0, 1, 4, and 5, the top-left
2 × 2 block. Patches go across a row first, then down, then to the next
frame. With more channels, each channel's four values come one after
another.

Each patch also carries a mask channel that marks the observed frames
(Section 4.6). So a patch holds `4 × (16 + 1) = 68` values. A linear layer
turns these 68 values into one token of 384 features.

## Ungroup Patches Back into the Latent

The model predicts a change for every latent cell, so its output must have
the latent's shape. The output layer gives each token 64 values: 4 cells ×
16 channels. [`unpatchify_output`](../../src/world_models/models/cosmos_transformer.py)
puts each value back in its cell. The output layer lists its values by cell
first and channel last, so this function ungroups in that order.

Next, we let the 320 tokens share information.
