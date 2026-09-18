# 3.7 Chapter Project: Build an Evaluation Report

Now we will bring the scores and our observations together in a report. The
goal is to answer one question: **would you choose guidance 7 or guidance 3
for the examples you evaluated, and why?**

## Collect the Results

We will compare guidance 7 and guidance 3 for the 14 PAI-Bench-G examples
from Section 3.5. Both use seeds 0 and 1. All other generation settings are
the same.

Use your results from Section 3.5 or the files linked below.

* **Videos:** open the [video gallery](../../../assets/chapter_03/pai/)
  to find both guidance settings for each example and seed.
* **Video quality scores:** [guidance 7](../../../assets/chapter_03/results/pai_guidance7_quality.json)
  and [guidance 3](../../../assets/chapter_03/results/pai_guidance3_quality.json),
  from the [Section 3.5 supplement](05-evaluate-with-pai-bench.md#supplement-measure-video-quality).
* **Benchmark questions and answers:** [guidance 7](../../../assets/chapter_03/results/pai_guidance7_vqa_detailed.json)
  and [guidance 3](../../../assets/chapter_03/results/pai_guidance3_vqa_detailed.json).

## Compare the Videos

Here are a few scores from Section 3.5. The Domain Score averages how often
Qwen's answers match the expected answers within each example, then across
examples. The quality scores help us compare appearance alongside those checks.

| Measurement | Guidance 7 | Guidance 3 |
|---|---:|---:|
| Domain Score | 0.8824 | 0.8231 |
| Subject consistency | 0.9188 | 0.9327 |
| Imaging quality | 0.7328 | 0.7288 |

Guidance 3 scores higher on subject consistency. Guidance 7 scores higher
on imaging quality and Domain Score. Let's watch the videos
to see whether we notice these differences.

In the gallery, choose the same example and seed at both guidance settings.
Check whether the requested event happens and whether the objects keep
their appearance as they move. Save timestamps for anything that helps
explain a score, including moments where you disagree with the evaluator.

## Write the Report

Keep the report short. Answer these four questions:

* **What did we compare?** Name the guidance settings and the examples and seeds
  you reviewed.
* **What did the scores show?** Include the measurements that matter for your
  choice.
* **What did we see?** Describe two or three examples from the videos, with
  timestamps so someone else can check them.
* **Which setting would we choose, and why?** Explain what matters most for
  your application and how the scores and observations support your choice.
  If you do not see a clear winner, explain why.

Keep your conclusion specific to these 14 examples. Save the report with the
scores and video references so another reader can check your reasoning.

## Keep the Evaluation Details

Attach these details so someone else can repeat the evaluation:

* **Versions and settings.** Record the generation model, judge, and
  evaluator versions and settings. Keep the benchmark questions and Qwen's
  answers from the
  [question-answering evaluator](https://github.com/SHI-Labs/physical-ai-bench/blob/2f3b687410029b98397fbc51fa4de36bfd45627d/generation/evaluate_vqa.py).
* **Video coverage.** Record the expected and actual video counts, plus
  missing or unreadable videos, evaluator errors, and failed or blocked
  generation runs. Leaving out difficult examples can make scores look better.
* **Video selection.** If you chose the best video from several attempts,
  record how many attempts, how you chose, and the generation cost.

When comparing with published results, follow the benchmark's evaluation
procedure. Report any changes to the model, judge, prompts, or score
calculation.
