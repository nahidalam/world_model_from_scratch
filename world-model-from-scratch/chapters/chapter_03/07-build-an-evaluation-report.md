# 3.7 Chapter Project: Build an Evaluation Report

Now we will bring the scores and our observations together in a report. The goal is to answer one question: **would you choose guidance 7 or guidance 3 for the cases you evaluated, and why?**

## Collect the Results

We will compare guidance 7 and guidance 3 for the 14 PAI-Bench-G scenarios from Section 3.5. Both use seeds 0 and 1. All other generation settings are the same.

Use your results from Section 3.5 or the files linked below.

* **Videos:** open the [video gallery](../../../assets/chapter_03/pai/) to find both guidance settings for each case and seed.
* **Video quality scores:** [guidance 7](../../../assets/chapter_03/results/pai_guidance7_quality.json) and [guidance 3](../../../assets/chapter_03/results/pai_guidance3_quality.json).
* **Benchmark questions and answers:** [guidance 7](../../../assets/chapter_03/results/pai_guidance7_vqa_detailed.json) and [guidance 3](../../../assets/chapter_03/results/pai_guidance3_vqa_detailed.json).

## Compare the Videos

Here are a few scores from Section 3.5. The last row measures how often Qwen's answers about the Cosmos videos matched the benchmark's expected answers.

| Measurement              | Guidance 7 | Guidance 3 |
| ------------------------ | ---------: | ---------: |
| Subject consistency      |     0.9188 |     0.9327 |
| Imaging quality          |     0.7328 |     0.7288 |
| Question-answer accuracy |     0.8824 |     0.8231 |

Guidance 3 scores higher on subject consistency. Guidance 7 scores higher on imaging quality and question-answer accuracy. Let's watch the videos to see whether we notice these differences.

In the gallery, choose the same case and seed at both guidance settings. Check whether the requested event happens and whether the objects keep their appearance as they move. Save timestamps for anything that helps explain a score, including moments where you disagree with the evaluator.

## Write the Report

Keep the report short. Answer these four questions:

* **What did we compare?** Name the guidance settings and the cases and seeds you reviewed.
* **What did the scores show?** Include the measurements that matter for your choice.
* **What did we see?** Describe two or three examples from the videos, with timestamps so someone else can check them.
* **Which setting would we choose, and why?** Explain what matters most for your use case and how the scores and observations support your choice. If you do not see a clear winner, explain why.

Keep your conclusion specific to these 14 cases. Save the report with the scores and video references so another reader can check your reasoning.
