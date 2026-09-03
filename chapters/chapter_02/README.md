# Chapter 2: Running a Pretrained World Model

In this chapter, we will use a pretrained video world model to generate future
video frames. The input is a sequence of observed video frames and a text prompt. The output
is a sequence of generated frames showing one possible continuation.

The examples use
[NVIDIA Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) model through Diffusers to generate video continuations from video and text.

## What You Will Build

By the end of the chapter, you will be able to:

* prepare a video observation for a released checkpoint
* load the checkpoint and generate one rollout
* sample several futures from the same observation
* compare guidance and inference-step settings
* measure disagreement among sampled futures
* save each rollout with the settings that produced it

## Contents

* [2.1 From an Observation to a Sampled Future](01-running-a-pretrained-world-model.md)
* [2.2 Choose a Setup](02-choose-a-setup.md)
* [2.3 Prepare the Observation](03-prepare-the-observation.md)
* [2.4 Load the Model and Generate a Rollout](04-load-and-generate.md)
* [2.5 Generate Several Futures](05-generate-several-futures.md)
* [2.6 Compare Generation Settings](06-control-and-inspect.md)
* [2.7 Save a Rollout with Its Settings](07-save-and-reuse.md)
