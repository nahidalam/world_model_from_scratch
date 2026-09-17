# 2.3 Prepare the Observation

We want to prepare our input for inference with Cosmos. It uses five consecutive frames as visual context for this rollout. The
running example takes the final five frames from
`assets/chapter_02/sand_mining.mp4`.

Load the video and select those frames:

```python
from world_models.observation import load_video

observation = load_video("assets/chapter_02/sand_mining.mp4")
context = observation.last(5)
print(context.frames.shape)
```

The output is:

```text
(5, 704, 1280, 3)
```

This five-frame observation is the context passed to the model in Section 2.4.
