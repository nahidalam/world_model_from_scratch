# 4.6 Condition on Observations

Our model needs to generate the video. Given frames 0 to 4, it generates
frames 5 to 16. To do that, it needs to know which latent frames are given
and which it must create. A mask marks them: `1` for the two observed latent
frames and `0` for the three to generate.

![Observed latent frames form a fixed prefix; a binary channel identifies them beside the noisy frames to generate.](../../figures/chapter_04/conditioning.svg)

*Figure 4.6: The five observed frames become two observed latent frames
(blue), and the three frames to generate start as noise (orange). A mask
marks observed positions with 1, and we concatenate it with the latent as a
17th channel.*

`prefix_mask` builds the mask for a latent of our shape:

```python
import torch
from world_models.small_world import prefix_mask

latent = torch.randn(1, 16, 5, 16, 16)
mask = prefix_mask(latent, observed_frames=2)
print(mask[0, 0, :, 0, 0])
print(torch.cat([latent, mask], dim=1).shape)
```

```text
tensor([1., 1., 0., 0., 0.])
torch.Size([1, 17, 5, 16, 16])
```

The mask has one value per frame. Joined to the latent, it adds a 17th
channel, which gives the 68 values per patch from Section 4.2.

The model uses the mask in four places:

* Model input. The observed frames sit in the first two latent positions,
  and the mask travels with them as the 17th channel.
* Noise level. The observed frames hold no noise, so `SmallWorldModel`
  gives them a noise level of `0.0001`, as the
  [Cosmos pipeline](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py)
  does. The other frames get `t`.
* Loss. The observed frames are known, so `flow_matching_loss` counts only
  the frames to generate.
* Sampler. Section 4.7 restores the observed frames before every step, so
  they stay equal to what we observed.

Next, we write the loop that runs the model from noise to a finished
video.
