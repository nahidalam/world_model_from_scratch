# Chapter 1 explorables

Self-contained interactive demos. Each is a single HTML file with inline CSS
and JavaScript: open it in any browser from a clone of the repository. No
build step, no install, no network access.

All four are live computations. The "world" in each is a hand-written rule
cheap enough to recompute on every interaction; no machine learning runs in
the browser.

| File | Section | What you manipulate |
|---|---|---|
| `transition_playground.html` | 1.1 | One predicted transition, a rollout, and multiple possible futures |
| `anticipation_gridworld.html` | 1.2 | Compare a distance rule with a planner |
| `simulator_vs_model.html` | 1.3 | Compare specified equations with a function fitted to samples |
| `pipeline_walkthrough.html` | 1.4 | Follow information through the world-model pipeline |
Colors follow the book's figure conventions: blue for what is learned, orange
for what is given, aqua for uncertainty.
