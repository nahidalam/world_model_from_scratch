# 2.7 Save a Rollout with Its Settings

A generated video does not show which model, prompt, or settings produced it.
Save that information alongside the rollout so the result can be inspected
later.

Generate and save the same configuration used in Section 2.4:

```python
from pathlib import Path

runs_dir = Path("outputs/chapter_02/runs")
saved_future = engine.generate(
    context,
    prompt=prompt,
    seed=0,
    num_frames=29,
    guidance_scale=7.0,
    num_inference_steps=15,
    save_to=runs_dir,
)

run_dir = runs_dir / saved_future.manifest.run_id
print(run_dir / "rollout.mp4")
print(run_dir / "manifest.json")
```

The output paths have this form:

```text
outputs/chapter_02/runs/<run_id>/rollout.mp4
outputs/chapter_02/runs/<run_id>/manifest.json
```

The manifest records the checkpoint revision, observation path, prompt, seed,
frame count, guidance, inference steps, output resolution, package versions,
and video path.

Chapter 3 will use saved rollouts and their manifests to evaluate prediction
quality.
