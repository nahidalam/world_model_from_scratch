# 3.3 Prepare Saved Runs for Evaluation

When we compare scores, we need to know which video each score belongs to and how that video was generated. Before running VBench, we will organize the saved Chapter 2 videos into an index. The index links each video to its seed, guidance value, inference steps, and saved run manifest, so we can trace every score back to the settings that produced the video.

![](../../.gitbook/assets/evaluation_flow.svg)

_Figure 3.1: We organize the saved videos and their generation settings in an index, run VBench on the videos, and collect the scores into a report for comparing runs._

## Manifest for Video Generation

Chapter 2 saved a `RunManifest` beside each generated video. The manifest for the seed-0 rollout looks like this:

```json
{
  "checkpoint": "nvidia/Cosmos-Predict2.5-2B",
  "revision": "diffusers/base/post-trained",
  "seed": 0,
  "num_frames": 29,
  "guidance_scale": 7.0,
  "num_inference_steps": 15,
  "height": 704,
  "width": 1280,
  "prompt": "an aerial view of a sand mining operation; the machinery keeps moving and the water keeps flowing",
  "observation": "assets/chapter_02/sand_mining.mp4",
  "outputs": [
    "assets/chapter_02/seeds/rollout_seed0.mp4"
  ],
  "versions": {
    "torch": "2.6.0+cu118",
    "diffusers": "0.39.0",
    "cosmos_guardrail": "not installed"
  },
  "created_at": "2026-08-16T06:45:18.354164+00:00",
  "run_id": "cbc346ed085d"
}
```

Load this manifest in Python to read its run ID and settings:

```python
from pathlib import Path

from world_models.experiment import RunManifest

manifest_path = Path("assets/chapter_02/seeds/rollout_seed0.manifest.json")
manifest = RunManifest.load(manifest_path)

print(manifest.run_id)
print(manifest.seed, manifest.guidance_scale, manifest.num_inference_steps)
```

When we prepare the videos, we will also calculate a checksum for each file. This lets us verify that VBench evaluates the exact video we prepared.

## Prepare the Videos for VBench

We will evaluate the four Chapter 2 rollouts generated with seeds 0, 1, 2, and 3. To prepare these saved videos for VBench, run the following command from the repository root:

```bash
python scripts/chapter_03_evaluate.py prepare \
  --assets assets/chapter_02/seeds \
  --output outputs/chapter_03/vbench
```

The command uses each selected manifest to find its video. It reads the frame count, frame rate, and resolution, then calculates the checksum. It also copies the videos without changing their frames or encoding. The prepared files look like this:

```
outputs/chapter_03/vbench/
  index.json
  videos/
    <run_id>.mp4
```

`index.json` records the source manifest, generation settings, video metadata, and checksum for each copied video.

## Inspect the Index

Before we run the evaluator, let's check that the index contains the videos and settings we expect. Read the index and find the baseline rollout: seed 0, guidance 7, and 15 inference steps.

```python
import json
from pathlib import Path

index_path = Path("outputs/chapter_03/vbench/index.json")
index = json.loads(index_path.read_text())

print("videos:", len(index["entries"]))

baseline = next(
    entry for entry in index["entries"]
    if entry["manifest"]["seed"] == 0
    and entry["manifest"]["guidance_scale"] == 7.0
    and entry["manifest"]["num_inference_steps"] == 15
)

print("run:", baseline["run_id"])
print("video:", baseline["video"])
print("settings:", baseline["manifest"]["seed"],
      baseline["manifest"]["guidance_scale"],
      baseline["manifest"]["num_inference_steps"])
print("media:", baseline["media"])
print("checksum:", baseline["video_sha256"])
```

The code prints:

```
videos: 4
run: cbc346ed085d
video: videos/cbc346ed085d.mp4
settings: 0 7.0 15
media: {'num_frames': 29, 'fps': 16.0, 'height': 704, 'width': 1280, 'duration_seconds': 1.8125}
checksum: 9fa8dbd7482f7ddb9aa25b3b3e0e1e0a08cdc28fa0f6d64250d4889c697381dc
```

> **Try it yourself**
>
> 1. Find the entry for seed 1 and compare its run ID and checksum with seed 0.
> 2. Check whether all four videos have the same frame count, frame rate, and resolution.
