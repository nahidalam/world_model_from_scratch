# 3.6 Check the Scores Against the Videos

A low score can mean Cosmos generated the wrong event, or Qwen misread the
video. We will watch the tennis-ball videos to check what happened.

## Check Qwen's Answer

For the tennis-ball example, `physics_002`, PAI-Bench-G asks:

> Does the tennis ball bounce off the coffee table before rolling off-screen?

The expected answer is **No**. Qwen answered **Yes** for both guidance-7
seeds, so this question scored 0.0. The
[evaluation results](../../assets/chapter_03/results/pai_guidance7_vqa_detailed.json)
contain the question and both answers.

Here are frames from the seed-1 video. They show no clear bounce or exit.
The ball remains on the tabletop in the last frame shown.

![Twelve frames from the guidance-7 seed-1 tennis-ball video, sampled every 0.5 seconds.](../../assets/chapter_03/results/physics_002_guidance7_seed1_contact_sheet.jpg)

These frames are 0.5 seconds apart, so they may miss a brief event. Watch the
[full seed-1 video](../../assets/chapter_03/pai/guidance7/videos/physics_002__1.mp4)
and answer the same question. Record the time range you checked.

* If the video shows the ball bouncing off the table before rolling out of
  view, Qwen's answer is correct. Cosmos failed this benchmark check.
* If the video clearly does not show that event, Qwen's answer is wrong.
  The score penalizes the video because of an error by Qwen.

If you cannot tell, record **Unclear**. The sampled frames alone are not
enough to decide.

## Compare Guidance 7 and Guidance 3

Qwen's bounce answer changes for seed 1 when we change guidance. Open both
videos:

| Guidance | Seed-1 video | Qwen's answer |
|---|---|---|
| 7 | [Watch video](../../assets/chapter_03/pai/guidance7/videos/physics_002__1.mp4) | Yes |
| 3 | [Watch video](../../assets/chapter_03/pai/guidance3/videos/physics_002__1.mp4) | No |

The starting image, prompt, seed, and other generation settings are the
same. Only guidance changed. The
[guidance-3 results](../../assets/chapter_03/results/pai_guidance3_vqa_detailed.json)
record Qwen's **No** answer, which matches the expected answer.

Watch both videos and check for the bounce. Does the changed answer reflect
a change in the ball's motion? If both videos show the same behavior for
this question, the different answers do not establish that one video is
better. Record what you see in each video and the relevant timestamps.

## Explain the Result

For each video, write down what happened, whether you agree with Qwen, and
whether the event matches the benchmark's expected answer. Include the
video name and timestamps so another reader can check your observation.
Keep the evaluator's original scores and record any disagreement separately.

Passing the bounce check does not tell us whether the ball completed the
requested action. It could stay on the table without bouncing or leaving
the frame. Check the questions about emerging from the pipe and rolling
out of view too.

Use these observations in the [evaluation report](07-build-an-evaluation-report.md).
They help explain which scores reflect the generated behavior and which
answers need further review.
