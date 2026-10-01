# Chapter 4: Building a World Model from Scratch

In this chapter, we will build a small world model that can be trained on a consumer GPU. This will be a video world model that continues an excavator's arm motion. We input five video frames and ask the model to generate the next twelve.

The model needs to learn how the scene changes over time. We use a small Transformer to learn these changes through attention across video patches. This follows the Cosmos design introduced in Chapter 2: encode the video, generate future latents with a Transformer, and decode them into frames.

We train the Transformer and its context vector from random weights. The pretrained WAN video autoencoder (VAE) compresses and reconstructs the video. We keep its weights fixed throughout training.

## Training on One GPU

We will train the model on a single consumer NVIDIA GPU with 8 GB of memory. The base model has 21.2 million parameters and trains on 17-frame clips at 128 × 128 pixels.

We encode each video once and save its latents before training. The training loop loads the Transformer and batches of these cached latents. Encoding, training, and decoding run separately to keep GPU memory use low.

## What We Will Build

Our world model takes five observed frames and predicts the next twelve. We build it in the order the data flows through it:

1. Turn video into tokens (4.2). A frozen VAE compresses the clip, and patches turn the latent into 320 tokens the Transformer can read.
2. Let tokens share information (4.3). Attention lets the bucket's token read the arm's pose and the bucket's earlier positions.
3. Decide what to predict (4.4). Flow matching trains the model to move a noisy latent a small step toward a clean one.
4. Build the predictor (4.5). A small Transformer reads the noise level at every layer and predicts that step.
5. Tie the prediction to the observation (4.6). The observed frames stay fixed, so the model continues this video rather than any video.
6. Generate a future (4.7). A sampling loop repeats the prediction from noise to a finished video.
7. Train and evaluate (4.8, 4.9). We train on 512 generated clips, then measure the bucket's path on unseen clips against holding the last frame still.

Our training scene is a simplified excavator moving its arm above sand. This is similar to the sand mining example we had in chapter 2 but more simplified to train in one consumer GPU. The tracked base stays in place while the arm raises and lowers its bucket. We generate the frames and save the bucket's position at each step, giving us both training data and a way to measure future motion.

## Setup

Run commands from the repository root. On the GPU machine, install a CUDA build of PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/) first. Then install the chapter dependencies and local package:

```bash
python -m pip install -r requirements-chapter-04.txt
python -m pip install -e .
```

Downloading the VAE uses the model access and Hugging Face authentication steps in [Chapter 2 setup](../chapter_02/02-choose-a-setup.md). The training path loads the WAN VAE directly and uses a learned context vector for the Transformer. The optional [released-checkpoint extension](a-reproduce-released-checkpoint.md) uses the full [Chapter 2 GPU environment](../chapter_02/02-choose-a-setup.md).

Start with the CPU check:

```bash
python scripts/chapter_04_train.py smoke \
  --device cpu \
  --output outputs/chapter_04/cpu_smoke
```

This runs a tiny latent training example and checks optimizer updates and checkpoint resume. Sections 4.8 and 4.9 provide the commands for preparing videos, training, and evaluating on a GPU. The [architecture explorer](https://github.com/nahidalam/world_model_from_scratch_private/tree/main/interactive/chapter_04/model_explorer.html) lets you inspect patches, attention weights, and noise mixtures in the browser.

## Contents

* [4.1 Choose the Model and the World](01-open-up-cosmos.md)
* [4.2 Turn Video into Latent Tokens](02-video-to-latent-tokens.md)
* [4.3 Implement Video Attention](03-implement-video-attention.md)
* [4.4 Implement Flow Matching](04-flow-matching.md)
* [4.5 Build the Small Transformer](05-build-the-transformer.md)
* [4.6 Condition on Observations](06-condition-on-observations.md)
* [4.7 Write the Sampling Loop](07-write-the-sampling-loop.md)
* [4.8 Train a Small World Model](08-train-a-small-world-model.md)
* [4.9 Evaluate the Trained Model](09-evaluate-the-trained-model.md)
* [4.10 The Complete Model in One File](10-the-complete-model.md)
* [Optional 4.A: Reproduce a Released Cosmos Checkpoint](a-reproduce-released-checkpoint.md)
