# Chapter 4 Verification Record

This page records the settings and saved measurements for the excavator
training example and the optional Cosmos checkpoint reproduction. We can
use these results to check our implementation and compare experiments.
The linked reports retain the hardware, software, and source details for
each run. Runtime depends on the device and configuration.

## Excavator Training: September 29, 2026

The scene has a fixed camera and tracked chassis, an articulated arm, and
an orange-red bucket over sand. Each seed selects a starting phase and
cycle speed. The task covers arm kinematics with fixed machine geometry.

Clips contain 17 frames at 128 × 128 pixels and 8 fps. Five observed frames
condition twelve future frames. The frozen WAN VAE produces a
`[16, 5, 16, 16]` latent, giving 320 Transformer tokens.

| Setting | Eight-clip fit | Full experiment |
|---|---:|---:|
| Preset | `small` | `base` |
| Trainable parameters | 7,818,240 | 21,164,160 |
| Training clips | 8 | 512 |
| Microbatch | 2 | 4 |
| Gradient accumulation | 4 | 8 |
| Effective batch | 8 | 32 |
| Optimizer updates | 2,000 | 2,000 |
| Learning rate | `3e-4` | `3e-4` |
| Training seed | 17 | 17 |

Each experiment uses one GPU. The environment uses Python 3.10.12,
PyTorch 2.6.0+cu124, Diffusers 0.39.0, Accelerate 1.14.0, NumPy 2.2.6,
and Pillow 12.3.0. Training uses
FP32 parameters and AdamW state with BF16 autocast. Activation checkpointing
is off in these measurements.

## Data and Compression

The full cache contains 512 training, 64 validation, and 64 test trajectories.
The split seeds start at 0, 1,000,000, and 2,000,000. The fitting cache uses
the first eight seeds in each split. The VAE comes from
`nvidia/Cosmos-Predict2.5-2B`, revision
`0d37c7498f54cee3c599d438d895a0a4a8608064`, subfolder `vae`.

Preparing all 640 clips took 70.16 seconds. The prefix check
encoded five frames independently and compared them with the first two
latents of the complete video. Its maximum absolute error was
0.0 for the first trajectory in each split.
The [cache record](../../assets/chapter_04/training/cache.json) stores the
split identities, source hashes, and measurements.

Across eight validation clips, frozen-VAE reconstruction gave
100% bucket detection, 0.183-pixel mean future
center error, and 0.02130 pixel RMSE on `[0, 1]` values.
The [reconstruction report](../../assets/chapter_04/training/reconstruction_validation.json)
and [paired video](../../assets/chapter_04/training/reconstruction/trajectory_1000000/reconstruction.mp4)
show the compression reference. All eight paired reconstructions are saved.

Reproduce the reconstruction check from the prepared cache:

```bash
python scripts/chapter_04_check_reconstruction.py \
  --data outputs/chapter_04/excavator/data \
  --output outputs/chapter_04/excavator/reconstruction_check \
  --device cuda
```

## Fitting and Full Training

The eight-clip run trained for 500 updates, then resumed to 2,000.
Its fixed training-probe loss fell from 2.38104 to
0.01387. Generating from those eight training observations
produced 0.435-pixel mean future bucket error, compared with
10.621 pixels for persistence, with 100% detection.
The [fit evaluation](../../assets/chapter_04/training/overfit_evaluation.json)
measures the complete encode, train, sample, and decode path.

The base model completed 2,000 optimizer updates in
927.36 seconds (15.46 minutes),
with 0 skipped updates. Its mean update time after warmup was
0.453 seconds.

| Fixed-noise probe | Before training | After training |
|---|---:|---:|
| Training loss | 2.38551 | 0.01018 |
| Validation loss | 2.37152 | 0.02862 |

The [training report](../../assets/chapter_04/training/training.json) and
[loss history](../../assets/chapter_04/training/metrics.jsonl) record the
configuration and optimizer updates.

Using the 2,000-update checkpoint and 30 Euler steps, all 64 validation
trajectories gave mean future bucket error 0.590 pixels, compared with
11.593 pixels for persistence. Detection coverage was
100.00%. The
[validation report](../../assets/chapter_04/results/base_training_validation.json)
records this check before test evaluation.

## Unseen Test Trajectories

The same checkpoint and 30-step sampler generate all 64 test futures.
Sampling seeds run from 123 through 186. Metrics cover the twelve future
frames of each clip:

| Metric | Generated future | Repeat last observed frame |
|---|---:|---:|
| Mean future bucket error | 0.591 px | 11.677 px |
| Final bucket error | 0.997 px | 20.516 px |
| Future pixel RMSE, `[0, 1]` | 0.03407 | 0.09361 |
| Bucket detection coverage | 100.00% | 100.00% |

The model improves mean bucket-position error on 64 of 64 trajectories.
The detector tracks the largest connected red region and checks its area
against the rendered bucket. Missing detections receive an image-diagonal
penalty. The [test report](../../assets/chapter_04/training/evaluation.json)
retains per-frame scores and detection coverage.

The published examples are the first test trajectory and those with the
lowest, upper-median, and highest mean position error: seeds
2,000,000, 2,000,010, 2,000,061, 2,000,033. The
[publication record](../../assets/chapter_04/training/publication.json)
records the selection rule and file hashes. Position scores measure the
bucket path; comparison videos also show the arm connections and machine
shape.

## GPU Memory and Runtime

These measurements describe the saved training experiment:

| Stage | Peak allocated | Peak reserved | Measured time |
|---|---:|---:|---:|
| Prepare 640 clips | 0.417 GiB | 0.520 GiB | 70.16 s |
| Fit eight clips, 2,000 updates | 0.277 GiB | 0.289 GiB | 359.71 s |
| Train base model, 2,000 updates | 0.917 GiB | 0.945 GiB | 927.36 s |
| Sample 64 test futures | 0.132 GiB | 0.152 GiB | 31.98 s |
| Decode 64 test clips | 0.426 GiB | 0.516 GiB | 9.50 s |

These are PyTorch CUDA allocation and reservation peaks. CUDA context and
driver memory add overhead. Encoding, training, sampling, and decoding run
as separate stages. Their measured footprints support the chapter's 8 GB
consumer GPU memory budget. We measure runtime on our own GPU to determine
how long each stage takes there.

Training time includes the optimizer loop, scheduled validation, and
checkpoint writes. Sampling and decoding times cover their named stages.
Process startup and media export add to command runtime.

## Implementation Checks

The repository suite passed 155 tests in both the local and GPU environments.
It checks excavator geometry and timing, bucket detection, cache causality,
optimizer updates, sampling, media export, and run manifests.

| Check | Result | Evidence |
|---|---|---|
| Exact CUDA resume | Four continuous updates match two updates followed by two resumed updates | [Resume report](../../assets/chapter_04/training/resume_verification.json) |
| FP16 training | Four updates, 0 skipped; peak reserved memory 0.945 GiB | [FP16 report](../../assets/chapter_04/results/fp16_training_check.json) |
| CPU optimizer and resume | 40 updates on synthetic latents; resumed parameter maximum error 0.0 | [CPU report](../../assets/chapter_04/results/training_smoke.json) |

The CUDA resume check compares model, optimizer, scaler, progress, and random
states. It uses deterministic CUDA settings and math attention. Regular
training uses PyTorch's selected attention kernels for performance.

Reproduce the CUDA resume check:

```bash
python scripts/chapter_04_verify_resume.py \
  --data outputs/chapter_04/excavator/data \
  --output outputs/chapter_04/excavator/resume_check \
  --device cuda \
  --preset small
```

The source archive, selected rollout manifests, and publication file hashes
preserve the implementation behind these results. Each saved latent file
contains one clip. The preview, reconstruction pairs, learning curves,
and selected rollout comparisons were visually inspected.

## Saved Experiment

The publisher gathers reports, plots, and selected videos from a completed
experiment. The book's artifacts live in `assets/chapter_04/training`.
With the saved run files under `outputs/chapter_04/`, we can rebuild the
artifact bundle:

```bash
chapter4_run=outputs/chapter_04/excavator_validation_20260929
python scripts/chapter_04_publish_training.py \
  --training "$chapter4_run/base_train" \
  --evaluation "$chapter4_run/base_test" \
  --overfit-training "$chapter4_run/overfit_train" \
  --overfit-evaluation "$chapter4_run/overfit_evaluation" \
  --cache "$chapter4_run/data" \
  --reconstruction "$chapter4_run/reconstruction_check" \
  --resume "$chapter4_run/resume_deterministic" \
  --output outputs/chapter_04/published_excavator_training
```

Choose a fresh publication directory. The publisher checks source, cache,
checkpoint, and rollout identities before copying artifacts and plotting
results.

## Archived Released-Checkpoint Experiment

The following source versions and measurements describe the September 9,
2026 checkpoint reproduction. Its scope is the full pretrained model and
the Chapter 2 sand-mining observation.

### Source Versions

* Model: `nvidia/Cosmos-Predict2.5-2B`, `diffusers/base/post-trained` release.
* Checkpoint revision: `0d37c7498f54cee3c599d438d895a0a4a8608064`.
* Reference implementation: Diffusers 0.39.0.
* Architecture: 28 blocks, width 2,048, 16 attention heads, 128 features per head.
* Paper: [World Simulation with Video Foundation Models for Physical AI, v1](https://arxiv.org/abs/2511.00062v1).

Table 3 of paper v1 lists 32 blocks for 2B. The implementation follows the
released checkpoint's 28-block configuration.

### Checks

The checkpoint experiment uses Python 3.10.12, PyTorch 2.6.0+cu124, Diffusers 0.39.0,
Transformers 5.14.1, Accelerate 1.14.0, and Cosmos Guardrail 0.3.1.

| Check | Result | Saved evidence |
|---|---|---|
| Repository tests, local CPU | 78 passed | `python -m pytest -q` |
| Repository tests, GPU host | 78 passed | `outputs/chapter_04/logs/tests.log` |
| Random-weight forward and backward pass | 35,376 parameters; loss 3.204773; 49 finite gradient tensors | [smoke.json](../../assets/chapter_04/results/smoke.json) |
| Five-frame VAE encode/decode | `[1, 3, 5, 704, 1280]` → `[1, 16, 2, 88, 160]` → original shape; reconstruction RMSE 0.046323 in normalized pixel values | [tokenizer.json](../../assets/chapter_04/results/tokenizer.json) |
| Full checkpoint mapping | All 569 tensors match; 2,059,174,912 parameters | [verify.json](../../assets/chapter_04/results/verify.json) |
| First block, last block, and velocity parity | Maximum absolute error 0.0 at noise levels 0.05, 0.50, and 0.95 | [verify.json](../../assets/chapter_04/results/verify.json) |

The archived component experiment can be run from the repository root:

```bash
python -m pytest -q
python scripts/chapter_04_experiments.py smoke --device cpu --output outputs/chapter_04/smoke
python scripts/chapter_04_experiments.py tokenizer --device cuda --output outputs/chapter_04/tokenizer
python scripts/chapter_04_experiments.py verify --device cuda --output outputs/chapter_04/verify
```

The checkpoint comparison uses BF16, a `[1, 16, 3, 4, 6]` latent tensor, and
synthetic text features. It asserts absolute and relative tolerances of 0.02;
the measured differences were zero. This run took 2.47 seconds after Python
startup and peaked at 3.90 GiB of allocated GPU memory. The VAE experiment
took 8.59 seconds and peaked at 9.40 GiB.

The verification covers patch order, attention, position encoding, timestep
modulation, conditioning masks, the velocity target, gradients, checkpoint
mapping, and intermediate activation agreement.

### Generated Rollouts

Generation uses the chapter's sand-mining observation. Both implementations
receive the same prompt, observation, random seed, frame count, guidance,
and scheduler settings.

* Input: the last five frames of `assets/chapter_02/sand_mining.mp4`.
* Prompt: `an aerial view of a sand mining operation; the machinery keeps moving and the water keeps flowing`.
* Seed 0; guidance 7; 15 sampling steps; 29 output frames at 704 × 1,280; 16 fps.
* Scheduler: the checkpoint's `UniPCMultistepScheduler`, with flow prediction
  and its saved sigma configuration.
* Frozen VAE and text encoder; model CPU offload enabled; safety checks active.

| Implementation | Run ID | Elapsed seconds | Peak allocated GPU memory |
|---|---|---:|---:|
| Diffusers reference | `f06ed3eddeb7` | 318.58 | 16.06 GiB |
| Chapter 4 Transformer and sampler | `648db6831993` | 303.13 | 11.44 GiB |

Elapsed time includes model loading, text preparation, generation, safety
checks, and export, after Python imports. The cards also served other jobs.
The runs used different offload scheduling, and the scratch run reused cached
text features. The differences in offload scheduling and cached text features are part of
the execution conditions. The timings and memory peaks apply to those
individual runs.

The latent tensors saved after updates 1, 8, and 15 match exactly. All
29 decoded frames also match exactly before MP4 encoding. Both MP4 files have
SHA256 `966eba9b2bea90ffdc38c465bf2a6bd3e4a768a71baea0745e8cf9c8d256f6e5`.
The [comparison report](../../assets/chapter_04/results/comparison.json)
records zero maximum error and zero RMSE for each saved tensor and the pixels.

The comparison checks the input checksum, generation settings, scheduler,
implementation labels, and report identities before comparing tensors.
[Optional Section 4.A](a-reproduce-released-checkpoint.md) contains the generation and
comparison commands, videos, and a frame comparison. The later frames show
the same changes in ground and water texture in both outputs. Numerical agreement establishes that our
code reproduces this reference run; Chapter 3's measurements address the
quality of the continuation.

### Artifact Locations

Runtime outputs live under `outputs/chapter_04`. The book includes selected
results under `assets/chapter_04`; the run manifests identify their source
checkpoint and experiment settings.

* [Reference report](../../assets/chapter_04/results/reference_generation.json)
  and [manifest](../../assets/chapter_04/results/reference_manifest.json).
* [Our report](../../assets/chapter_04/results/scratch_generation.json)
  and [manifest](../../assets/chapter_04/results/scratch_manifest.json).
* [Reference video](../../assets/chapter_04/reference_rollout.mp4),
  [our video](../../assets/chapter_04/scratch_rollout.mp4), and
  [frame comparison](../../assets/chapter_04/rollout_comparison.png).
* [VAE reconstruction video](../../assets/chapter_04/tokenizer_reconstruction.mp4)
  and [frame comparison](../../assets/chapter_04/tokenizer_reconstruction.png).
* [Exact generation source snapshot](../../assets/chapter_04/results/generation_source.zip).

The source archive preserves the four files named by the generation reports'
SHA256 values. We can use these checksums to identify the exact implementation
that produced the saved results.

To rebuild the book's result bundle from the saved runtime outputs:

```bash
unzip -p assets/chapter_04/results/generation_source.zip \
  scripts/chapter_04_experiments.py > outputs/chapter_04/generation_source.py
python scripts/chapter_04_publish_results.py \
  --generation-script outputs/chapter_04/generation_source.py
```

Git ignores the runtime `outputs/chapter_04` directory. The selected book
artifacts, source archive, and manifests remain under `assets/chapter_04`.
