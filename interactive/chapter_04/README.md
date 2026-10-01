# Chapter 4 explorer

Open [model_explorer.html](model_explorer.html) in a browser. All HTML, CSS,
JavaScript, and illustrative input values live in the file. It needs no build
step, network connection, Python environment, or GPU.

The page computes three examples:

1. Pack two 4 × 4 single-channel frames into eight 2 × 2 patches. Select a token
   to locate its source values. The page unpacks the tokens and reports whether
   all 32 values match the input.
2. Compute attention over the eight patch positions. Choose a query and toggle
   coordinate rotations. Inspect scores, normalized weights, and the weighted
   value. Queries and keys use six handcrafted features, three rotation pairs,
   and RMS normalization. Values are patch means.
3. Mix a 16 × 16 illustrative latent with seeded Gaussian noise. Change flow
   time, noise seed, or the inspected cell. Read the interpolation and velocity
   target as numerical calculations.

These examples compute the tensor operations on illustrative arrays. The
chapter's Python training lab applies the same operations to a small video
Transformer and measures the futures it generates.

Link directly to [patch selection](model_explorer.html#tokens),
[attention](model_explorer.html#attention), or
[flow interpolation](model_explorer.html#flow). Frame dimensions and the number
of observed frames are fixed in these examples.

The chapter's [architecture figures](../../figures/chapter_04/) follow the
21.2M-parameter training model: 17 frames at 128 × 128 pixels become 320 tokens
with 384 features each. Five observed pixel frames supply two known latent
slices; the model learns to generate the remaining three latent slices. A
frozen WAN VAE prepares the cache and decodes samples. The Transformer reads
one shared learned context token and uses Euler steps to generate the future.

The optional [released Cosmos architecture](../../figures/chapter_04/cosmos_reference_architecture.svg)
documents the full pretrained reference model. Regenerate all figures from the
repository root:

```bash
python scripts/chapter_04_build_figures.py
```

The figure generator uses the Python standard library.
