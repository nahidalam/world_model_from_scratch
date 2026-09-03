# Chapter 2 Rollout Explorer

`rollout_explorer.html` is a replay of the Chapter 2 artifacts. It does not run
Cosmos in the browser. Each chapter section links directly to its relevant
view; the views are not intended as a four-step activity at the start of the
chapter.

The page has four views:

1. `#observation` compares the final context frame with one sampled future for
   Sections 2.1–2.4.
2. `#seeds` synchronizes four rollouts generated with different seeds for
   Section 2.5.
3. `#controls` compares guidance and inference-step settings for Section 2.6.
4. `#disagreement` compares a source rollout with the disagreement map for
   Section 2.6.

The page uses repository-relative files from `assets/chapter_02/`. It contains
no external scripts or network requests.
