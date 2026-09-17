# 3.2 Existing Evaluation Benchmarks

In this chapter, we will use VBench to measure consistency and motion, and
PAI-Bench-G to evaluate what happens in a generated video.

| Evaluation question | Benchmark | Measurement |
|---|---|---|
| Does the subject keep the same appearance? | VBench | Subject consistency |
| Does movement remain smooth? | VBench | Motion smoothness |
| Does the video contain movement? | VBench | Dynamic degree |
| Does the requested action occur? | PAI-Bench-G | Question-answer accuracy |
| Do objects maintain the expected spatial relationship? | PAI-Bench-G | Question-answer accuracy |
| Does the event show the expected physical behavior? | PAI-Bench-G | Question-answer accuracy |

## Measure Consistency and Motion with VBench

[VBench](https://arxiv.org/abs/2311.17982) divides video evaluation into 16
dimensions, each measuring one property, such as subject consistency,
motion smoothness, or how closely the video follows the prompt.

We will run three dimensions on four Chapter 2 seed rollouts:

* **Subject consistency** measures whether the subject keeps a similar
  appearance across frames.
* **Motion smoothness** measures whether movement continues without sudden
  jumps between frames.
* **Dynamic degree** checks whether the video contains enough movement to
  count as dynamic.

We can apply these measurements to our saved sand-mining videos using VBench's
[`custom_input` mode](https://github.com/Vchitect/VBench#new-evaluate-your-own-videos).
We will report one result for each of the three dimensions. To run a full
VBench evaluation, we would need its complete prompt set and all required
dimensions.

## Evaluate the Requested Event with PAI-Bench-G

[PAI-Bench](https://arxiv.org/abs/2512.01989) has three tracks:

* **PAI-Bench-G** evaluates video generation.
* **PAI-Bench-C** checks whether videos follow control signals such as depth
  maps, edges, and segmentation masks.
* **PAI-Bench-U** evaluates how models understand physical scenes.

We will use PAI-Bench-G with the included videos, which were generated from
its case images and prompts.

A physics case, for example, may ask whether an object keeps its shape while
falling.

The [PAI-Bench-G evaluation workflow](https://github.com/SHI-Labs/physical-ai-bench/tree/2f3b687410029b98397fbc51fa4de36bfd45627d/generation)
also measures video quality, including subject consistency, background
consistency, motion smoothness, and alignment with the starting image.

## Other Benchmarks

The right benchmark depends on what the model predicts. We are evaluating
videos here, but other world models predict weather fields, physical states,
or the result of an action. The following benchmarks cover these different
outputs and the questions we might ask about them.

| Benchmark | Domain | What it evaluates |
|---|---|---|
| [VBench-2.0](https://vchitect.github.io/VBench-2.0-project/) | Video generation | Human appearance, composition, physics, and common sense |
| [VideoPhy](https://arxiv.org/abs/2406.03520) | Video generation | Material interactions and physical rules |
| [WorldModelBench](https://arxiv.org/abs/2502.20694) | Video world models | Instruction following and physical behavior |
| [WorldSimBench](https://proceedings.mlr.press/v267/qin25f.html) | Embodied systems | Video quality and support for control tasks |
| [WeatherBench 2](https://arxiv.org/abs/2308.15560) | Weather | Global weather forecasts compared with atmospheric data |
| [ClimateBench](https://doi.org/10.1029/2021MS002954) | Climate | Temperature and precipitation responses to emission scenarios |
| [PDEBench](https://arxiv.org/abs/2210.07182) | Physical simulation | Predictions for systems governed by partial differential equations |
| [PHYRE](https://arxiv.org/abs/1908.05656) | Physical reasoning | Actions that solve mechanics puzzles in a 2D environment |

If you want to explore more benchmarks, [A Survey of AI-Generated Video Evaluation](https://arxiv.org/abs/2410.19884)
and [A Comprehensive Survey on World Models for Embodied AI](https://arxiv.org/abs/2510.16732)
are good starting points.

Next, we will prepare the four seed rollouts for VBench.
