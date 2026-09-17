# 3.5 Evaluate with PAI-Bench-G

In Section 3.4, we used VBench to measure consistency and motion in the sand-mining videos from Chapter 2. Now we will use [PAI-Bench-G](https://arxiv.org/html/2512.01989v1#S3.SS1) to evaluate video quality and check whether the events described in the prompts actually happen.

Note that PAI-Bench-G provides a starting image, a prompt, and questions for each case. Those questions are specific to the scene. For example, a question about a tennis ball rolling off a table would not apply to our sand-mining videos. To use the benchmark's questions, we need videos generated from the corresponding images and prompts.

We used Cosmos-Predict2.5 to generate these videos in advance from the benchmark's inputs. They are included in this book's repository under `assets/chapter_03/pai/`, so we can focus on evaluating them in this chapter.

For this evaluation, we selected two cases from each of the benchmark's seven categories. This gives us 14 cases. For each case, we generated videos with seeds 0 and 1 at both guidance 7 and guidance 3.

Let's start with the 28 videos generated at guidance 7. The [PAI-Bench-G evaluators](https://github.com/SHI-Labs/physical-ai-bench/blob/2f3b687410029b98397fbc51fa4de36bfd45627d/generation/README.md) give us two sets of measurements:

| Evaluation      | Input                                       | Output                                                                         |
| --------------- | ------------------------------------------- | ------------------------------------------------------------------------------ |
| Video quality   | Generated video, starting image, and prompt | Eight quality scores, including subject consistency and motion smoothness      |
| Requested event | Generated video and case questions          | Answers and accuracy for actions, spatial relationships, and physical behavior |

Next, we will evaluate the same cases and seeds at guidance 3. The [video gallery](../../../assets/chapter_03/pai/) contains all 56 videos with their generation settings. If you do not have a GPU, use the saved results below to compare the scores with the videos.

## Download the Images, Prompts, and Questions

To evaluate each video, we need the starting image and prompt it was generated from, along with the benchmark's questions. The `prepare` command downloads these for our 14 cases. Run it from the book's repository root:

```bash
BOOK_ROOT="$(pwd)"
PAI_SUBSET="$BOOK_ROOT/outputs/chapter_03/pai/subset"
PAI_VIDEOS="$BOOK_ROOT/assets/chapter_03/pai/guidance7/videos"
PAI_RECORD="$BOOK_ROOT/assets/chapter_03/results/pai_guidance7_generation.json"
PAI_RUN="$BOOK_ROOT/outputs/chapter_03/pai/guidance7"

python -m pip install huggingface_hub
python scripts/chapter_03_generate_pai.py prepare \
  --per-category 2 \
  --output "$PAI_SUBSET"
```

The command downloads a fixed version of the [PAI-Bench-G dataset](https://huggingface.co/datasets/shi-labs/physical-ai-bench-generation/tree/c36b7beb69bd7c4adcedc1431407fce98f9dfe44) into this directory:

```
subset/
  subset.json
  full_info.json
  condition_image/
  vqa/
```

The files contain:

* `full_info.json`: the prompts.
* `condition_image/`: the starting images.
* `vqa/`: the benchmark questions.
* `subset.json`: the selected cases and dataset version.

The book also includes a copy of the [case list](../../../assets/chapter_03/results/pai_subset.json) and [prompts](../../../assets/chapter_03/results/pai_subset_full_info.json).

Let's look at one example. The code below prints the first prompt and the paths to its starting image and questions:

```bash
python - "$PAI_SUBSET" <<'PY'
import json
import sys
from pathlib import Path

subset = Path(sys.argv[1])
cases = json.loads((subset / "full_info.json").read_text())
first = cases[0]

print("cases:", len(cases))
print("first case:", first["video_id"])
print("image:", subset / "condition_image" / first["image_name"])
print("prompt:", first["prompt_en"])
print("questions:", subset / "vqa" / f"{first['video_id']}.json")
PY
```

Open the image and question file. Read the prompt, then check what the questions ask about the video.

Before running the evaluators, let's check that each of our 14 cases has a video for seed 0 and seed 1. The command below should find 28 videos:

```bash
python - "$PAI_VIDEOS" "$PAI_RECORD" "$PAI_SUBSET" <<'PY'
import json
import sys
from pathlib import Path

videos, record, subset = map(Path, sys.argv[1:])
plan = json.loads(record.read_text())
cases = json.loads((subset / "full_info.json").read_text())
if {case["video_id"] for case in cases} != set(plan["video_ids"]):
    raise SystemExit("Downloaded cases do not match the saved videos")
expected = {
    f"{case}__{seed}.mp4"
    for case in plan["video_ids"] for seed in plan["seeds"]
}
actual = {path.name for path in videos.glob("*.mp4")}

if actual != expected:
    raise SystemExit(
        f"Missing: {sorted(expected - actual)}; extra: {sorted(actual - expected)}"
    )

print(f"Complete: {len(actual)} videos for {len(plan['video_ids'])} cases")
PY
```

The command prints:

```
Complete: 28 videos for 14 cases
```

## Set Up the Evaluators

We will install the evaluators in their own environment so that [PAI-Bench-G's dependencies](https://github.com/SHI-Labs/physical-ai-bench/blob/2f3b687410029b98397fbc51fa4de36bfd45627d/generation/pyproject.toml), including vLLM and Detectron2, do not conflict with our VBench setup. Run these commands on a Linux CUDA machine:

```bash
PAI_ROOT="$BOOK_ROOT/outputs/chapter_03/tools/physical-ai-bench"
PAI_EVALUATOR="$BOOK_ROOT/outputs/chapter_03/pai/evaluator"
mkdir -p "$BOOK_ROOT/outputs/chapter_03/tools" "$PAI_EVALUATOR" \
  "$PAI_RUN/evaluation"
python -m pip install uv
git clone https://github.com/SHI-Labs/physical-ai-bench.git "$PAI_ROOT"
git -C "$PAI_ROOT" checkout 2f3b687410029b98397fbc51fa4de36bfd45627d
cd "$PAI_ROOT/generation"

uv sync
uv pip install --python "$PAI_ROOT/generation/.venv/bin/python" \
  --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"

# vLLM 0.14 rejects expert parallelism for this dense Qwen checkpoint.
sed -i \
  's/"enable_expert_parallel": True/"enable_expert_parallel": False/' \
  "$PAI_ROOT/generation/pbench/vqa_evaluation.py"

git rev-parse HEAD > "$PAI_EVALUATOR/evaluator_revision.txt"
git -C "$PAI_ROOT" diff -- generation/pbench/vqa_evaluation.py \
  > "$PAI_EVALUATOR/vqa_vllm_compat.patch"
uv pip freeze --python "$PAI_ROOT/generation/.venv/bin/python" \
  > "$PAI_EVALUATOR/evaluator_environment.txt"
cp uv.lock "$PAI_EVALUATOR/uv.lock"
"$PAI_ROOT/generation/.venv/bin/python" - "$PAI_EVALUATOR" <<'PY' \
  > "$PAI_EVALUATOR/judge_snapshot.txt"
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

revision = "89c86200743eec961a297729e7990e8f2ddbc4c5"
snapshot = snapshot_download("Qwen/Qwen2.5-VL-72B-Instruct", revision=revision)
(Path(sys.argv[1]) / "judge_revision.txt").write_text(revision + "\n")
print(snapshot)
PY

cp "$PAI_EVALUATOR"/* "$PAI_RUN/evaluation/"
```

The setup downloads Qwen to answer the benchmark's questions. It also records the model, code, and package versions so we can repeat the evaluation. The [vLLM compatibility patch](../../../assets/chapter_03/results/pai_vqa_vllm_compat.patch) shows the one-line change needed to run Qwen with our version of vLLM.

## Measure Video Quality

First, let's measure the quality of our 28 guidance-7 videos. The [quality evaluator](https://github.com/SHI-Labs/physical-ai-bench/blob/2f3b687410029b98397fbc51fa4de36bfd45627d/generation/evaluate.py) checks eight properties, including clarity, consistency, and motion. Run it with the videos, prompts, and starting images:

```bash
(
  cd "$PAI_ROOT/generation" || exit 1
  uv run python -m torch.distributed.run \
    --standalone --nproc_per_node 1 evaluate.py \
    --mode custom_input \
    --prompt_file "$PAI_SUBSET/full_info.json" \
    --custom_image_folder "$PAI_SUBSET/condition_image" \
    --dimension aesthetic_quality background_consistency imaging_quality \
      motion_smoothness overall_consistency subject_consistency \
      i2v_background i2v_subject \
    --videos_path "$PAI_VIDEOS" \
    --output_path "$PAI_RUN/evaluation/quality"
)
```

The evaluator saves results for individual videos and for the full set. When it finishes, find the result files with:

```bash
find "$PAI_RUN/evaluation/quality" -name 'results_*.json' -print
```

Here are the aggregate scores from our guidance-7 run. All 28 videos were scored on each of the eight dimensions:

| Dimension                 | Aggregate |
| ------------------------- | --------: |
| Aesthetic quality         |    0.5133 |
| Background consistency    |    0.9377 |
| Imaging quality           |    0.7328 |
| Motion smoothness         |    0.9896 |
| Overall consistency       |    0.2195 |
| Subject consistency       |    0.9188 |
| Image-to-video background |    0.9710 |
| Image-to-video subject    |    0.9602 |

You can find each video's scores in the [quality results](../../../assets/chapter_03/results/pai_guidance7_quality.json).

## Check Whether the Requested Events Happen

In PAI-Bench-G, Qwen was used as the question-answer model. Therefore, we will use Qwen to check whether the requested events happen in the Cosmos videos.

[question-answering evaluator](https://github.com/SHI-Labs/physical-ai-bench/blob/2f3b687410029b98397fbc51fa4de36bfd45627d/generation/evaluate_vqa.py) compares those answers with the expected answers.

The command below runs `Qwen/Qwen2.5-VL-72B-Instruct` across eight GPUs:

```bash
(
  cd "$PAI_ROOT/generation" || exit 1
  uv run python evaluate_vqa.py \
    --prompt_file "$PAI_SUBSET/full_info.json" \
    --vqa_questions_dir "$PAI_SUBSET/vqa" \
    --video_dir "$PAI_VIDEOS" \
    --model_name "$(cat "$PAI_EVALUATOR/judge_snapshot.txt")" \
    --device cuda --tensor_parallel_size 8 \
    --output_dir "$PAI_RUN/evaluation/vqa"
)
```

The evaluator creates two files:

* `vqa_summary.json`: the overall score and scores for each category.
* `vqa_detailed_results.json`: each question, its expected answer, Qwen's answers for both seeds, and the question's accuracy.

Check that both files are present:

```bash
ls "$PAI_RUN/evaluation/vqa/vqa_summary.json" \
   "$PAI_RUN/evaluation/vqa/vqa_detailed_results.json"
```

The 14 scenarios we selected from PAI-Bench-G contain 111 questions. The evaluator uses those questions and calculates the scores in three steps:

1. For each question, score Qwen's answers for seeds 0 and 1. Average the two scores.
2. Average the question scores within each case.
3. Average the case scores to get the overall score. Do the same within each category to get its score.

Open `vqa_detailed_results.json` and choose a question with a low score. Watch both seed videos to check whether you agree with Qwen's answers.

Here are the question-answering scores for the guidance-7 videos:

| Scope              | Accuracy |
| ------------------ | -------: |
| Overall            |   0.8824 |
| Autonomous driving |   0.5833 |
| Common sense       |   1.0000 |
| Human activity     |   0.9333 |
| Industry           |   1.0000 |
| Miscellaneous      |   1.0000 |
| Physics            |   0.8434 |
| Robotics           |   0.8167 |

You can find our results in the [summary](../../../assets/chapter_03/results/pai_guidance7_vqa_summary.json) and [individual answers](../../../assets/chapter_03/results/pai_guidance7_vqa_detailed.json).

## Repeat the Evaluation with Guidance 3

We have evaluated the Cosmos videos generated with guidance 7. Now let’s evaluate the videos generated with guidance 3 to see how the scores change. We use the same 14 PAI-Bench-G scenarios with seeds 0 and 1. This gives us 28 videos for each guidance setting. All other generation settings remain the same.

```bash
PAI_VIDEOS="$BOOK_ROOT/assets/chapter_03/pai/guidance3/videos"
PAI_RECORD="$BOOK_ROOT/assets/chapter_03/results/pai_guidance3_generation.json"
PAI_RUN="$BOOK_ROOT/outputs/chapter_03/pai/guidance3"
mkdir -p "$PAI_RUN/evaluation"
cp "$PAI_EVALUATOR"/* "$PAI_RUN/evaluation/"
```

Repeat the video check, then run both evaluators with these paths. Use the same benchmark inputs and evaluator setup.

You can also use the guidance-3 [quality scores](../../../assets/chapter_03/results/pai_guidance3_quality.json), [question-answering summary](../../../assets/chapter_03/results/pai_guidance3_vqa_summary.json), and [individual answers](../../../assets/chapter_03/results/pai_guidance3_vqa_detailed.json). The [generation settings](../../../assets/chapter_03/results/pai_guidance3_generation.json) and [run index](../../../assets/chapter_03/results/pai_guidance3_run_index.jsonl) record how these videos were made and which files were evaluated.
