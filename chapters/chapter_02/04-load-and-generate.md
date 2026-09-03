# 2.4 Load the Model and Generate a Rollout

Readers with the GPU setup can generate the rollout below. The included figure
and explorer show the result without requiring local generation.

## Load the Pipeline

Load Cosmos-Predict2.5-2B through Diffusers:

```python
import torch
from diffusers import Cosmos2_5_PredictBasePipeline

from world_models.backends.cosmos import (
    CHECKPOINTS,
    CosmosBackend,
    apply_guardrail_compatibility_patch,
)
from world_models.engine import RolloutEngine

spec = CHECKPOINTS["cosmos-predict2.5-2b-720p"]
apply_guardrail_compatibility_patch()

pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
    spec.repo_id,
    revision=spec.revision,
    torch_dtype=torch.bfloat16,
).to("cuda")

engine = RolloutEngine(CosmosBackend(pipeline, spec))
```

`CosmosBackend` adapts the Diffusers pipeline to the frame format used in this
chapter. `RolloutEngine` provides the generation interface used below.

## Generate One Future

Use the five-frame `context` prepared in Section 2.3:

```python
prompt = (
    "an aerial view of a sand mining operation; "
    "the machinery keeps moving and the water keeps flowing"
)

future = engine.generate(
    context,
    prompt=prompt,
    seed=0,
    num_frames=29,
    guidance_scale=7.0,
    num_inference_steps=15,
)

print(future.frames.shape)
```

The output is:

```text
(29, 704, 1280, 3)
```

This rollout took about two minutes on the NVIDIA A40.

![](../../figures/chapter_02/filmstrip_seed0.png)

*Figure 2.1: Six generated frames sampled in temporal order from the first
rollout.*

[Open the first rollout in a new tab](https://nahidalam.github.io/world_model_from_scratch/interactive/chapter_02/rollout_explorer.html#observation).
