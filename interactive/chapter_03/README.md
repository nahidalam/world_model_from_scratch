# Chapter 3 Evaluation Explorer

Open `evaluation_explorer.html` in a browser from a repository clone. The page
replays existing Chapter 2 videos, so visual inspection and annotation require
no GPU, installation, external scripts, or network service. Keep the HTML inside
the repository so its relative video and manifest links resolve.

The chapter links to three sections:

- `#compare`: select two runs and use a shared frame slider or playback to inspect
  the same moment. The selectors include four seeds, guidance 3/7/12, and
  15/20/36/50 inference steps. Each selection displays its actual run ID and
  generation settings, with a link to the shipped manifest.
- `#scores`: import a local JSON evaluation report. The page displays scores only
  for matching run IDs, in the evaluator's original scale. No scores are bundled
  or simulated. Custom-video results remain distinct from full benchmark results.
- `#review`: answer questions about motion continuity, subject identity, and
  prompt adherence. Save evidence at a chosen frame and export a human-review
  JSON file. Notes remain in page memory until exported; reloading clears them.

Frame numbers start at zero and refer to the exported 29-frame clip at 16 fps,
including its conditioning region. The shared control seeks both videos to the
same frame. These Chapter 2 examples have no held-out recorded future and cannot
establish prediction accuracy.

The optional evaluation report is an object with a `rows` array. Each row must
contain a string `run_id`, a string `dimension`, and a finite numeric `score`.
The page also accepts a bare array of these rows. Metadata outside `rows`,
including `scope` and evaluator provenance, appears under Report provenance.
Import the report produced by the chapter evaluation script after running an
official evaluator. The browser does not execute an evaluator.

Run metadata is embedded from the shipped `assets/chapter_02/` manifests so the
explorer works when opened directly from disk. The guidance-7 baseline and the
seed-0 baseline have the same run ID. The 15-step option uses that existing
baseline video. The page does not imply that new clips were generated for
Chapter 3.

The human-review export uses `kind: "human_review"` and an `annotations` array,
separate from evaluator score rows. Each annotation records the run ID, video
and manifest paths, frame index, timestamp, question, judgment, and evidence.
