# 2.6 Compare Generation Settings

Section 2.5 changed the seed while keeping the other settings fixed. Here we
keep the seed fixed and compare two generation settings: guidance and inference
steps.

## Guidance

`guidance_scale` determines how strongly generation follows the text prompt.
The explorer compares values 3, 7, and 12:

```python
guidance_runs = [
    engine.generate(
        context,
        prompt=prompt,
        seed=0,
        num_frames=29,
        guidance_scale=guidance,
        num_inference_steps=15,
    )
    for guidance in (3.0, 7.0, 12.0)
]
```

Compare how closely each rollout follows the prompt, along with its motion,
continuity, and visual artifacts.

## Inference Steps

`num_inference_steps` controls how many refinement steps the model performs
while generating the video. The explorer compares 15, 20, 36, and 50 steps:

```python
step_runs = [
    engine.generate(
        context,
        prompt=prompt,
        seed=0,
        num_frames=29,
        guidance_scale=7.0,
        num_inference_steps=steps,
    )
    for steps in (15, 20, 36, 50)
]
```

More steps use more computation but do not increase the number of output frames;
every rollout in this comparison still contains 29 frames. Compare the videos
instead of assuming that the largest value is best.

[Open the generation-settings view in a new tab](https://nahidalam.github.io/world_model_from_scratch/interactive/chapter_02/rollout_explorer.html#controls).
