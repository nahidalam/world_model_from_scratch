# 2.1 From an Observation to a Sampled Future

In this section, we will use the pretrained world model [NVIDIA Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) to generate a rollout.

For each rollout, Cosmos-Predict2.5 receives a sequence of frames and a text
prompt. It outputs a sequence of frames. The input will not have any action sequence in this case. Therefore the output is a probable continuation of the observed scene, not the consequence of
a chosen action.

Use the explorer to compare the final observed frame with one generated
rollout. Move the frame control through the generated continuation.

[Open the Chapter 2 rollout explorer in a new tab](https://nahidalam.github.io/world_model_from_scratch/interactive/chapter_02/rollout_explorer.html#observation).
