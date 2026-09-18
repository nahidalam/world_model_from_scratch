# Chapter 3 Validation Record

Checked on 2026-09-09. The GPU runs used eight NVIDIA A40 GPUs with 48 GB of
memory each. The book includes the compact evaluator results linked from
Sections 3.4 and 3.5. On September 16, the original PAI videos and their
provenance records were copied from the GPU run directories into the
[matched-video gallery](../../assets/chapter_03/pai/README.md). All 56 video
checksums match the previously saved run indexes.

## September 16 Review Corrections

* Generation now accepts an immutable `--checkpoint-revision`, defaulting to
  the recorded Cosmos SHA. Regression checks verify that the same revision
  reaches the model loader, generation record, and run manifest.
* Both PAI evaluator commands establish their working directory in a subshell.
  Revised Qwen setup downloads a fixed snapshot and loads the model,
  processor, and tokenizer from that path.
* VBench records the exact result path from each invocation, so older
  timestamped files cannot expand into extra collector arguments.
* The video-quality supplement in Section 3.5 reproduces imaging-quality
  normalization and the reported guidance difference from the included
  per-video results.
* The gallery preserves 56 videos, 56 original manifests, and each run's
  evaluator environment, lockfile, revision records, and compatibility patch.
  Its `review_index.json` maps portable asset paths to run IDs and checksums.

Local verification for these corrections passed:

* 37 tests across generation-revision, evaluation, and manifest behavior,
  including nine new checks for revision propagation and invalid revisions.
* The documented VBench run and collection blocks executed twice against a
  local evaluator fixture with an older result present and spaces in paths.
* Both PAI evaluator blocks ran against command fixtures from the book root;
  they selected the correct working directory and local judge snapshot path.
* All 56 copied videos and 56 manifests match their indexed hashes, and all
  ten evaluator records match their source hashes. Every MP4 contains 93
  frames at 1280 × 704 and 16 fps.
* The normalization example matches its printed output. Chapter and gallery
  links resolve, and Bash and Python examples pass syntax checks.

To recreate the gallery from downloaded original run directories, run:

```bash
python scripts/chapter_03_prepare_review.py \
  --runs outputs/chapter_03/review_download \
  --output assets/chapter_03/pai
```

The preparation command validates the full source set before copying it and
refuses videos that differ from the previously scored pilot.

The archived evaluator records are unchanged. Their historical
`judge_revision.txt` records the Hub revision queried during the original
run; it does not independently prove which snapshot every loader used. The
revised instructions now bind the loader to the recorded snapshot explicitly.
The original scores were preserved; GPU generation and evaluation were not
rerun for these corrections.

## Completed Checks

| Check | Evidence | Result |
|---|---|---|
| Local Python tests | `.venv/bin/python -m pytest -q`, Python 3.12.12 | **44 passed**. Includes manifest, conditioning, video preparation, evaluator-result parsing, and the Cosmos guardrail compatibility fix. |
| VBench input preparation | Section 3.3 command using `assets/chapter_02/seeds` | **Passed**. The command staged four seed rollouts and recorded their settings, media metadata, and checksums in `index.json`. |
| VBench evaluation | VBench `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`; PyTorch 2.5.1+cu118 | **Passed**. All three dimensions returned one score for each of the four rollouts. The aggregate scores were 0.9389 subject consistency, 0.9929 motion smoothness, and 0 dynamic degree. |
| PAI subset preparation | `python scripts/chapter_03_generate_pai.py prepare --per-category 2 --output <new-directory>/subset` | **14 cases downloaded**, two per category, from the pinned dataset revision. The cases contain 111 questions. |
| Prepared-input validation | `load_prepared()` on the downloaded subset | **Passed**. The selected case IDs and all 29 recorded file checksums matched: 14 images, 14 question files, and `full_info.json`. |
| Conditioning-image decoding | `load_image()` on every selected image using the local book environment | **14 passed** as single-frame RGB `uint8` observations. This checks loading, not Cosmos inference. |
| Cosmos smoke generation | One case, seed 0, 29 frames, 15 steps, guidance 7 | **Passed**. The MP4 contains 29 frames and its manifest records the case, prompt, generation settings, checkpoint revision, and checksum. |
| PAI generation | Cosmos-Predict2.5 `0d37c7498f54cee3c599d438d895a0a4a8608064`; guidance 7 and 3; 14 cases; seeds 0 and 1; 93 frames; 36 steps | **Passed**. Eight shards produced 28 videos for each setting. Each merge checked complete case/seed coverage and every SHA-256 checksum. |
| PAI evaluator installation | PAI-Bench-G `2f3b687410029b98397fbc51fa4de36bfd45627d`; Detectron2 `a2f4a8771ab77e8411c26b27f24f9489a28a2453` | **Passed**. Each run preserves the evaluator commit, installed environment, lockfile, and Qwen checkpoint revision. |
| PAI quality evaluation | Eight official dimensions, one process on one A40 | **Passed** for guidance 7 and guidance 3. Each dimension contains all 28 per-video results. Sections 3.5 and 3.7 report the aggregates and link both result files. |
| PAI requested-event evaluation | Qwen2.5-VL-72B-Instruct `89c86200743eec961a297729e7990e8f2ddbc4c5`; tensor parallel size 8 | **Passed** for both settings. Each result contains 14 cases, 111 questions, and two answers per question. Guidance 7 scored 0.8824 overall; guidance 3 scored 0.8231. |
| Final output validation | Run-index and SHA-256 comparison; `ffprobe` over every MP4; evaluator-result coverage checks | **Passed**. Both 28-video grids match their run indexes and checksums. All 56 MP4 files contain 93 frames at 1280×704 and 16 fps. Every quality dimension contains 28 filenames, and both VQA files contain the planned 14 cases, 111 questions, and two seeds. |
| PAI source review | Official generation README, dependency specification, quality CLI/indexing, and VQA CLI/implementation at commit `2f3b687410029b98397fbc51fa4de36bfd45627d` | Commands, filename conventions, judge defaults, and aggregation behavior checked against source. Citations appear in Section 3.5. |
| Chapter navigation and example syntax | Local link targets in Chapter 3, `README.md`, and `SUMMARY.md`; `bash -n` and Python AST parsing of their examples | **Passed**: no missing local link targets, 20 Bash blocks and three Python blocks parsed. Syntax checking does not execute GPU commands. |

The prepared dataset revision is
`c36b7beb69bd7c4adcedc1431407fce98f9dfe44`. The selected case IDs are:

```text
av_002b815b-7595-4d54-a9f4-4c62c83d3009
av_00d8e9d4-d669-42d9-8aa9-16a09a47b531
common_sense_049c9292-db5f-490b-a72b-546775e8156c
common_sense_062f7b4e-1f09-4a1e-9f1a-5c531c9e2155
human_000
human_001
industry_001
industry_002
misc_1093662-hd_1920_1080_30fps
misc_1200903-hd_1920_1080_30fps
physics_001
physics_002
robot_000
robot_001
```

## VBench Baseline Selection

Executing `prepare_vbench(Path("assets/chapter_02"), output)` with the shipped
assets fails with:

```text
ValueError: Different video bytes share run_id cbc346ed085d
```

The following two files have the same recorded generation configuration and
run ID, but different SHA-256 checksums:

| Video | SHA-256 |
|---|---|
| `assets/chapter_02/seeds/rollout_seed0.mp4` | `9fa8dbd7482f7ddb9aa25b3b3e0e1e0a08cdc28fa0f6d64250d4889c697381dc` |
| `assets/chapter_02/guidance_sweep/guidance_7_seed0.mp4` | `4e541f35fb8deed9d17a661360f0f146cc782fabbc00894eec9231cbce1ca91b` |

The preparation code refuses to choose between these files silently. Section
3.3 prepares the four videos in the seed experiment, so this conflict does not
enter the VBench input set. The checksum check still rejects the unfiltered
Chapter 2 asset directory.

## Execution Notes

VBench writes colons in its timestamped filenames. The mounted output
filesystem rejected those names, so the evaluator wrote to `/tmp` and the
copy step replaced each colon with a hyphen.

The first Cosmos smoke run found that NLTK's hardened file reader rejected
the Hugging Face cache symlinks used by `cosmos_guardrail` 0.3.1. The Cosmos
backend now stages those tokenizer files as regular files before loading the
safety checker. The smoke run and both full generations passed with the
guardrail enabled.

Two quality-evaluator processes raced while populating a shared DINO cache and
then reached an NCCL timeout. The documented one-process command completed the
same eight official dimensions without changing their implementations.

The pinned PAI-Bench-G dependency permits `vllm>=0.10.1`; the installation
resolved vLLM 0.14.0. That release rejects the evaluator's expert-parallel
option for the dense Qwen checkpoint. We disabled that execution option and
saved the one-line source diff with each run. The checkpoint, prompts,
questions, answer check, seed aggregation, and eight-way tensor parallelism
remain unchanged.

Each 93-frame Cosmos video took between 25 minutes 14 seconds and 25 minutes
28 seconds on one A40. Eight shards generated different cases in parallel.
The guidance-7 and guidance-3 quality runs took 12 minutes 58 seconds and 8
minutes 22 seconds. The successful Qwen runs took 14 minutes 52 seconds and 9
minutes 54 seconds, including model loading.

For the manual result check, we sampled `physics_002` guidance-7 seed 1 every
0.5 seconds. The benchmark expected **no** for the question about the ball
bouncing off the table before rolling out of view; the judge answered **yes**.
The sampled frames show no clear bounce or exit, so Section 3.6 records this as
a judge disagreement that needs full-video review.
