# 3.1 What Are We Evaluating?

So what are we evaluating? We need to decide what makes a generated video
useful. A video might look visually appealing, but whether it is useful
depends on what we want to evaluate.

A single metric cannot tell us everything. For example, a stationary video
may receive a strong consistency score because nothing changes. But if our
prompt asks machinery to move and water to flow, we also need to check whether
that movement happens. We will look at several properties separately:

| Property | What to look for |
|---|---|
| Visual quality | Are the frames clear and free of distracting artifacts? |
| Consistency | Do objects keep the same appearance and identity over time? |
| Motion | Does movement continue smoothly instead of freezing or jumping? |
| Requested event | Does the video show the action or interaction described by the prompt? |
| Physical plausibility | Do objects move and interact in a believable way? |

These properties help us judge how the videos look and whether their events
are plausible. To check prediction accuracy, we would need a recording of
what actually happened next in the scene. Our Chapter 2 examples do not
include that recording. To evaluate an agent's decisions, we would also need
measures of task performance, such as success rate or return.

## Measurement with VBench

We can start with the sand-mining videos from Chapter 2. They all continue
from the same observation and prompt, so we can compare what changes when we
use different seeds, guidance values, or inference steps.

We will use [VBench](https://arxiv.org/abs/2311.17982) to put numbers alongside
our observations of three properties:

* subject consistency
* motion smoothness
* dynamic degree

## Evaluating with PAI-Bench-G

[PAI-Bench-G](https://arxiv.org/abs/2512.01989) examines the requested event.
Each case provides a starting image, a prompt, and questions about what the
video should show. These questions cover expected actions, spatial
relationships, and physical behavior. We will use them to evaluate the PAI
videos included with the book.

For example, the benchmark may ask, "Did the vehicle move into the left-turn
lane?" An evaluator watches the generated video and answers the question. The
benchmark compares that response with the expected answer.

## Activity: Compare Two Rollouts

Before we run an evaluator, we will record our own observations. Open the
[evaluation explorer](https://nahidalam.github.io/world_model_from_scratch/interactive/chapter_03/evaluation_explorer.html#compare)
and compare two Chapter 2 videos:

1. Select seed 0 for Video A and seed 1 for Video B. Both videos use guidance
   7 and 15 inference steps, so the seed is the only setting that changes.
2. Play the videos together and follow the same machine or patch of water in
   both. Pause when you see a difference in its shape or motion.
3. In **Record visible evidence**, choose one video and answer **Does motion
   stay continuous?** Describe the object, its location, and what changes near
   the selected frame. Save the annotation.
4. Repeat the observation for the other video, then select **Export review
   JSON**.

Open the exported `chapter_03_human_review.json` file. Each annotation keeps
your observation with the run ID, video and manifest, selected frame,
timestamp, question, and judgment. This lets us return to the moment you
described when we compare your observations with the VBench scores later.

Next, we will look at the available benchmarks and the questions each one can
answer.
