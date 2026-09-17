# Chapter 3: Evaluating World Models

In Chapter 2, we generated possible futures of a scene. In this chapter, we learn to evaluate those futures. We measure visual quality, continuity, and whether the requested events occur plausibly.

First, we will use [VBench](https://arxiv.org/abs/2311.17982) to evaluate the sand-mining videos saved in Chapter 2. This will help us measure how consistent the objects look and how smoothly they move.

Then we will use [PAI-Bench-G](https://arxiv.org/abs/2512.01989) to evaluate the PAI videos included with the book. We will check whether they show the requested actions, whether objects are in the expected positions relative to each other, and whether their physical behavior looks plausible.

## What You Will Build

* an overview of video world-model benchmarks with links to their sources
* a collection of saved videos prepared for an official evaluator
* a VBench comparison of four Chapter 2 seed rollouts
* a PAI-Bench-G evaluation of the included videos
* a report with per-video scores and timestamped visual observations

## Setup

We will use a Linux machine with an NVIDIA GPU to run the benchmark evaluators on the videos provided with this book. Sections [3.4](04-evaluate-with-vbench.md) and [3.5](05-evaluate-with-pai-bench.md) walk through installing the evaluators in separate environments. PAI-Bench-G's default question-answering judge uses multiple GPUs.

Use Python 3.10 or later and install the book's dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you do not have GPU access, use the included Chapter 2 videos and the [evaluation explorer](https://nahidalam.github.io/world_model_from_scratch/interactive/chapter_03/evaluation_explorer.html) to compare generation settings, record observations at specific frames, and export a qualitative review. This browser exercise requires no installation. You can also import the included [VBench report](../../../assets/chapter_03/results/vbench_report.json) to compare your observations with the saved scores.

For the PAI experiment, the [matched-video gallery](../../../assets/chapter_03/pai/) includes the 56 original scored videos, their manifests, and evaluator records. Review them alongside the saved quality and question-answering results without generating new videos.

## Contents

* [3.1 What Are We Evaluating?](01-what-are-we-evaluating.md)
* [3.2 Existing Evaluation Benchmarks](02-existing-benchmarks.md)
* [3.3 Prepare Saved Runs for Evaluation](03-prepare-saved-runs.md)
* [3.4 Evaluate Our Rollouts with VBench](04-evaluate-with-vbench.md)
* [3.5 Evaluate with PAI-Bench-G](05-evaluate-with-pai-bench.md)
* [3.6 Interpret Results](06-interpret-results.md)
* [3.7 Chapter Project: Build an Evaluation Report](07-build-an-evaluation-report.md)
