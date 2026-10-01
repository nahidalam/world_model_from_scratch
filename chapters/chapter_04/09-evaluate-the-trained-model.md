# 4.9 Evaluate the Trained Model

In the previous section, the loss fell on training and validation clips.
Here we check the videos: we generate futures for 64 test clips and measure
how far the bucket is from its true position.

## Test Rollouts

Running this command first with `--split val` and its own `--output` folder
picks the checkpoint and sampling settings. With those settings fixed, we
run the test clips once:

```bash
python scripts/chapter_04_train.py sample \
  --data outputs/chapter_04/excavator/data \
  --checkpoint outputs/chapter_04/excavator/train/checkpoint.pt \
  --output outputs/chapter_04/excavator/evaluation \
  --device cuda \
  --split test \
  --examples 64 \
  --sampling-steps 30
```

For each clip, the model receives the two observed latent frames, generates
three more, and decodes all 17 frames. Each rollout is saved with a
`RunManifest` recording its checkpoint, seed, and settings.

## Results

As a baseline, we repeat the last observed frame, which keeps the arm still.
A detector finds the bucket in each generated frame, and we measure its
distance from the true position: the mean over frames 5 to 16 and the
distance in frame 16. Pixel RMSE compares the whole image.

| Measurement | Generated future | Repeat last frame |
|---|---:|---:|
| Mean bucket error | 0.59 px | 11.68 px |
| Final bucket error | 1.00 px | 20.52 px |
| Pixel RMSE, values 0–1 | 0.0341 | 0.0936 |

The model beats the baseline on all 64 test clips
([test report](../../assets/chapter_04/training/evaluation.json)):

![Future bucket-center error for the generated video and persistence on each of 64 test trajectories.](../../assets/chapter_04/training/motion_error.png)

*Figure 4.9: Mean bucket error on each test clip, for the generated video
and for holding the last frame.*

![Ground truth, generated future, and persistence for the median excavator test sequence.](../../assets/chapter_04/training/examples/main/trajectory_2000061/comparison.png)

*Figure 4.10: The median example follows the bucket as the arm extends to
the right. Compare the connected arm segments and bucket position at each
time.*

Mean bucket error ranges from 0.17 pixels
([video](../../assets/chapter_04/training/examples/main/trajectory_2000010/rollout.mp4)) to 2.46 pixels
([video](../../assets/chapter_04/training/examples/main/trajectory_2000033/rollout.mp4)). Decoding the true latents
already gives 0.183 pixels, so part of the error comes from the frozen VAE.

The model continues the arm's motion on clips outside its training set,
with the bucket about half a pixel from its true path on average. Section
4.10 collects the whole model in one file.
