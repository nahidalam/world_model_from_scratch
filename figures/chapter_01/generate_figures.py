"""Generate every Chapter 1 figure.

Run from anywhere:

    python figures/chapter_01/generate_figures.py

Each function draws one figure from its specification in the chapter text and
writes a PNG next to this script. The drawing vocabulary (boxes, arrows,
accents) lives in ``src/world_models/diagrams.py`` and is shared across
chapters; blue marks what is learned, orange what is given, aqua what is
uncertain.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from world_models.diagrams import (  # noqa: E402
    STYLE,
    arrow,
    box,
    canvas,
    dot,
    label,
    panel,
    save,
    wedge,
)

OUT = Path(__file__).resolve().parent


def fig_1_1_transition() -> Path:
    """One-step prediction, often carried out through a learned representation."""
    from matplotlib.patches import Rectangle

    fig, ax = canvas(122, 48)

    for cx, title in ((14, "current state\nor observation"),
                      (108, "next state\nor observation")):
        box(ax, cx, 27, 18, 18, "")
        for i in range(4):
            for j in range(4):
                dot(ax, cx - 6 + i * 4, 20.5 + j * 3.8, 0.5, accent="neutral")
        label(ax, cx, 9.8, title, size=STYLE.label_size)

    for cx, title in ((40, "internal\nrepresentation"),
                      (82, "next\nrepresentation")):
        box(ax, cx, 27, 14, 14, "", accent="learned")
        for k, h in enumerate((3.8, 6.1, 2.7)):
            ax.add_patch(
                Rectangle(
                    (cx - 4.5 + k * 3.2, 21.1),
                    2.0,
                    h,
                    facecolor=STYLE.learned,
                    edgecolor="none",
                    zorder=3,
                )
            )
        label(ax, cx, 11.2, title, size=STYLE.small_size)

    box(ax, 61, 27, 16, 11.5, "world model", accent="learned")
    box(ax, 61, 11, 15, 8, "action", accent="given")

    arrow(ax, (23, 27), (33, 27), label="encode")
    arrow(ax, (47, 27), (53, 27))
    arrow(ax, (61, 15), (61, 21.0), accent="given")
    arrow(ax, (69, 27), (75, 27), label="predict", accent="learned")
    arrow(ax, (89, 27), (99, 27), label="decode")
    return save(fig, OUT / "fig_1_1_transition.png")


def fig_1_2_representation() -> Path:
    """Predict in the compact middle, not on raw pixels."""
    fig, ax = canvas(112, 46)
    # Busy raw observation: a frame filled with a grid of cells.
    for cx, name in ((14, "observation"), (98, "next observation")):
        box(ax, cx, 26, 20, 20, "")
        for i in range(4):
            for j in range(4):
                dot(ax, cx - 6 + i * 4, 19.5 + j * 4.3, 0.55, accent="neutral")
        label(ax, cx, 12.5, name, size=STYLE.label_size)
    # Compact internal representations: three bars each.
    for cx, name in ((45, "internal representation"), (67, "next representation")):
        box(ax, cx, 26, 13, 13, "", accent="learned")
        for k, h in enumerate((3.4, 5.4, 2.4)):
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (cx - 4.4 + k * 3.2, 21.8), 2.0, h,
                facecolor=STYLE.learned, edgecolor="none", zorder=3))
        label(ax, cx, 16.2, name, size=STYLE.small_size, wrap=16)
    arrow(ax, (24, 26), (38.5, 26), label="encode")
    arrow(ax, (51.5, 26), (60.5, 26), label="predict", accent="learned")
    arrow(ax, (73.5, 26), (88, 26), label="decode")
    label(ax, 56, 40.5, "prediction happens here, in the compact space",
          color=STYLE.learned, size=STYLE.small_size)
    return save(fig, OUT / "fig_1_2_representation.png")


def fig_1_3_many_futures() -> Path:
    """One state and action fan into several weighted next states."""
    fig, ax = canvas(100, 44)
    box(ax, 16, 22, 22, 10, "current state")
    ys, alphas = (36, 27, 17, 8), (0.95, 0.7, 0.45, 0.28)
    wedge(ax, [(27, 22), (68, 40), (68, 4)])
    for y, a in zip(ys, alphas):
        b = box(ax, 79, y, 26, 7.5, "possible next state", accent="spread",
                faded=a < 0.5, size=STYLE.small_size)
        arr = arrow(ax, (27, 22), (66, y), accent="spread", faded=a < 0.5)
    label(ax, 46, 24.6, "one action", size=STYLE.small_size, color=STYLE.ink_soft, style="italic")
    label(ax, 50, 3.2, "opacity suggests likelihood", size=STYLE.tiny_size, color=STYLE.ink_faint)
    return save(fig, OUT / "fig_1_3_many_futures.png")


def fig_1_4_reactive_vs_planning() -> Path:
    """The world model and planner as separate parts of model-based planning."""
    fig, ax = canvas(124, 44)

    box(ax, 10, 23, 16, 10, "current observation", accent="given",
        size=STYLE.small_size, wrap=12)
    box(ax, 34, 23, 19, 11, "Planner", sub="propose candidate plans",
        size=STYLE.label_size, wrap=14)
    box(ax, 63, 23, 21, 13, "World model", sub="predict rollouts",
        accent="learned", size=STYLE.label_size, wrap=14)
    box(ax, 92, 23, 19, 11, "Planner", sub="score the rollouts",
        size=STYLE.label_size, wrap=14)
    box(ax, 116, 23, 16, 10, "selected action", accent="given",
        size=STYLE.small_size, wrap=12)

    arrow(ax, (18, 23), (24.5, 23))
    arrow(ax, (43.5, 23), (52.5, 23), label="plans")
    arrow(ax, (73.5, 23), (82.5, 23), label="rollouts")
    arrow(ax, (101.5, 23), (108, 23))

    return save(fig, OUT / "fig_1_4_reactive_vs_planning.png")


def _future_tree(ax):
    """Shared tree for figures 1.5 and 1.6: one root, three actions, nine leaves."""
    root = (13, 26)
    box(ax, *root, 16, 9, "now")
    branches = {
        "brake": (44, 41),
        "hold": (44, 26),
        "swerve": (44, 11),
    }
    leaves = {
        "brake": [(79, 46, 0.9, False), (79, 40, 0.55, False), (79, 34.4, 0.3, False)],
        "hold": [(79, 28.6, 0.75, False), (79, 23, 0.5, True)],
        "swerve": [(79, 17, 0.7, False), (79, 11.4, 0.45, True), (79, 5.8, 0.25, False)],
    }
    for name, (bx, by) in branches.items():
        arrow(ax, (21, 26), (bx - 5.5, by), label=name, accent="given", label_offset=0.6)
        dot(ax, bx - 3.5, by, 0.8, accent="neutral")
        for lx, ly, a, costly in leaves[name]:
            arrow(ax, (bx - 3.5, by), (lx - 3.2, ly), accent="spread", faded=a < 0.5)
            dot(ax, lx, ly, 1.5, accent="spread", faded=a < 0.5)
            if costly:
                label(ax, lx + 3.4, ly, "costly", size=STYLE.tiny_size,
                      color=STYLE.given, ha="left")
    return branches, leaves


def fig_1_5_one_state_many_futures() -> Path:
    """Two robot paths produce different distributions over outcomes."""
    fig, ax = canvas(100, 52)

    box(ax, 14, 26, 20, 9, "robot at junction")
    branches = (
        ("left path", 43, 38,
         ((46, "goal in 4: 60%", False),
          (39, "goal in 5: 25%", False),
          (32, "trap: 15%", False))),
        ("right path", 43, 14,
         ((22, "goal in 7: 55%", False),
          (15, "goal in 8: 40%", False),
          (8, "trap: 5%", True))),
    )

    for name, bx, by, outcomes in branches:
        arrow(ax, (24, 26), (bx - 4, by), label=name, accent="given",
              label_offset=0.8)
        dot(ax, bx, by, 0.8, accent="neutral")
        for ly, text, faded in outcomes:
            arrow(ax, (bx + 1, by), (72, ly), accent="spread",
                  faded=faded)
            box(ax, 84, ly, 20, 5.5, text, accent="spread",
                faded=faded, size=STYLE.tiny_size)

    return save(fig, OUT / "fig_1_5_one_state_many_futures.png")


def fig_1_6_anticipation_better_decisions() -> Path:
    """Reactive and anticipatory choices on the same tree."""
    fig, ax = canvas(100, 52)
    branches, leaves = _future_tree(ax)
    # Reactive picks the branch that looks best right now and lands in a costly leaf.
    arrow(ax, (21, 26.8), (39, 27.2), accent="given", curve=-0.12)
    arrow(ax, (40.5, 26.6), (75, 23.4), accent="given", curve=-0.08)
    label(ax, 58, 32.6, "reactive (best now)", size=STYLE.tiny_size, color=STYLE.given)
    # Anticipatory inspects the leaves and takes the branch with the best outcome.
    arrow(ax, (21, 25.2), (39, 40.4), accent="learned", curve=0.12)
    arrow(ax, (40.5, 41.4), (75, 45.6), accent="learned", curve=0.10)
    label(ax, 55, 48.9, "anticipatory (best outcome)", size=STYLE.tiny_size,
          color=STYLE.learned)
    return save(fig, OUT / "fig_1_6_anticipation_better_decisions.png")


def fig_1_7_simulator_vs_world_model() -> Path:
    """Three types of simulator, separated by their source of dynamics."""
    fig, ax = canvas(120, 72)

    rows = (
        (56, "Rule-based simulator", "specified rules", "given"),
        (35, "Learned-model simulator", "learned world model", "learned"),
        (14, "Hybrid simulator", "rules + world model", "neutral"),
    )
    for y, title, dynamics, accent in rows:
        panel(ax, 60, y, 116, 18, title, accent=accent)
        box(ax, 16, y - 2, 19, 7.5, "state + action",
            size=STYLE.small_size)
        box(ax, 60, y - 2, 28, 7.5, dynamics, accent=accent,
            size=STYLE.small_size)
        box(ax, 104, y - 2, 19, 7.5, "next state",
            size=STYLE.small_size)
        arrow(ax, (25.5, y - 2), (46, y - 2))
        arrow(ax, (74, y - 2), (94.5, y - 2))
    return save(fig, OUT / "fig_1_7_simulator_vs_world_model.png")


def fig_1_8_complementary() -> Path:
    """Four ways the two systems feed each other."""
    fig, ax = canvas(104, 54)
    box(ax, 26, 34, 26, 12, "simulator", accent="given")
    box(ax, 78, 34, 26, 12, "world model", accent="learned")
    arrow(ax, (39, 37.5), (65, 37.5), label="synthetic data", accent="given")
    arrow(ax, (65, 31), (39, 31), label="fast surrogate", accent="learned",
          label_offset=-3.4)
    box(ax, 26, 9, 22, 8.5, "policy", size=STYLE.small_size)
    arrow(ax, (26, 28), (26, 13.5), label="evaluation bench", accent="given",
          label_offset=0.2)
    ax.add_patch(__import__("matplotlib.patches", fromlist=["FancyBboxPatch"]).FancyBboxPatch(
        (58, 3.5), 40, 12, boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=STYLE.line_width, linestyle="--", edgecolor=STYLE.ink_soft,
        facecolor="none", zorder=1))
    label(ax, 78, 12.2, "hybrid", size=STYLE.small_size, color=STYLE.ink_soft, weight="bold")
    label(ax, 78, 7.6, "known physics as constraints, a learned residual for the rest",
          size=STYLE.tiny_size, color=STYLE.ink_soft, wrap=34)
    return save(fig, OUT / "fig_1_8_complementary.png")


def fig_1_9_two_failures() -> Path:
    """Omission versus extrapolation, as two maps of a state space."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = canvas(112, 54)
    panel(ax, 29, 26, 52, 48, "Simulator: rule coverage", accent="given")
    panel(ax, 85, 26, 52, 48, "World model: data coverage", accent="learned")
    # Left: a hard-edged modeled region with a gap it never covers.
    ax.add_patch(FancyBboxPatch((12, 12), 30, 24, boxstyle="round,pad=0,rounding_size=2",
                                linewidth=2.0, edgecolor=STYLE.given,
                                facecolor=STYLE.given_fill, zorder=1))
    label(ax, 27, 26, "modeled region", size=STYLE.small_size, color=STYLE.ink)
    ax.add_patch(FancyBboxPatch((34, 15), 12, 8, boxstyle="round,pad=0,rounding_size=1.4",
                                linewidth=1.4, linestyle="--", edgecolor=STYLE.ink_soft,
                                facecolor=STYLE.surface, zorder=2))
    label(ax, 40, 19, "fog: not\nmodeled", size=STYLE.tiny_size, color=STYLE.ink_soft)
    label(ax, 29, 7.4, "outside the rules: unspecified", size=STYLE.tiny_size,
          color=STYLE.ink_faint)
    # Right: accuracy fading outward from the data.
    for grow, a in ((16, 0.10), (10, 0.16), (4, 0.30)):
        ax.add_patch(FancyBboxPatch((85 - 12 - grow / 2, 24 - 7 - grow / 2),
                                    24 + grow, 14 + grow,
                                    boxstyle="round,pad=0,rounding_size=2",
                                    linewidth=0, facecolor=STYLE.learned,
                                    alpha=a, zorder=1))
    label(ax, 85, 24, "training data\nregion", size=STYLE.small_size)
    label(ax, 85, 9.0, "far from the data: less reliable", size=STYLE.tiny_size,
          color=STYLE.learned)
    return save(fig, OUT / "fig_1_9_two_failures.png")


def fig_1_10_blueprint() -> Path:
    """Inputs, internal operations, and prediction output."""
    fig, ax = canvas(116, 44)
    stages = (
        (14, "Observation", "input from the environment", "given"),
        (42, "Representation", "state used for prediction", "learned"),
        (70, "Dynamics", "advance the state through time", "learned"),
        (100, "Prediction", "output about the future", "learned"),
    )
    for x, name, sub, acc in stages:
        box(ax, x, 27, 24, 13, name, sub=sub, accent=acc, wrap=14)
    for x0, x1 in ((26, 30), (54, 58), (82, 87)):
        arrow(ax, (x0, 27), (x1, 27))
    box(ax, 70, 9, 24, 9, "Action", sub="input to the state update", accent="given", wrap=14)
    arrow(ax, (70, 13.5), (70, 20.5), accent="given")
    return save(fig, OUT / "fig_1_10_blueprint.png")


def fig_1_11_information_flow() -> Path:
    """Three unrolled steps: the advanced state becomes the next input."""
    fig, ax = canvas(116, 42)
    for i, x in enumerate((22, 58, 94)):
        box(ax, x, 24, 26, 11, f"step t+{i}", sub="the five components",
            accent="learned", wrap=16)
        box(ax, x - 5, 7.5, 13, 6.5, f"action", accent="given", size=STYLE.tiny_size)
        arrow(ax, (x - 5, 10.8), (x - 5, 18.2), accent="given")
        arrow(ax, (x + 6, 18.2), (x + 6, 12.6), accent="learned")
        label(ax, x + 6, 10.4, "prediction", size=STYLE.tiny_size, color=STYLE.learned)
    arrow(ax, (35, 26.5), (44.5, 26.5), label="state", accent="learned")
    arrow(ax, (71, 26.5), (80.5, 26.5), label="state", accent="learned")
    arrow(ax, (8, 32), (13, 28.4), label="observation", accent="given", label_offset=0.8)
    label(ax, 58, 38.6, "the same update pattern repeats; the state carries forward",
          size=STYLE.small_size, color=STYLE.ink_soft, style="italic")
    return save(fig, OUT / "fig_1_11_information_flow.png")


def fig_1_12_responsibility_table() -> Path:
    """The five components as a reference card."""
    fig, ax = canvas(104, 58)
    rows = (
        ("Observation", "brings in current information", "given"),
        ("Representation", "stores predictive state", "learned"),
        ("Action", "steers state evolution", "given"),
        ("Dynamics", "advances the state one step", "learned"),
        ("Prediction", "exposes usable outputs", "learned"),
    )
    label(ax, 52, 54.0, "The five components", size=STYLE.title_size, weight="bold")
    for i, (name, job, acc) in enumerate(rows):
        y = 45.0 - i * 8.6
        box(ax, 20, y, 30, 7.2, name, accent=acc, size=STYLE.label_size)
        label(ax, 38.5, y, job, ha="left", size=STYLE.small_size, color=STYLE.ink_soft)
    label(ax, 52, 3.0, "the same colors mark the same parts in every architecture diagram in the book",
          size=STYLE.tiny_size, color=STYLE.ink_faint)
    return save(fig, OUT / "fig_1_12_responsibility_table.png")


def fig_1_13_lifecycle_circle() -> Path:
    """Six stages as a closed ring with shortcut arrows."""
    import numpy as np

    fig, ax = canvas(96, 74, scale=0.105)
    cx, cy, r = 48, 37, 26
    stages = ("data collection", "data preparation", "training",
              "evaluation", "deployment", "continuous improvement")
    pos = []
    for i, name in enumerate(stages):
        th = np.pi / 2 - i * (2 * np.pi / 6)
        x, y = cx + r * np.cos(th), cy + r * np.sin(th) * 0.92
        pos.append((x, y))
        box(ax, x, y, 24, 8.5, name, accent="learned" if name == "training" else "neutral",
            size=STYLE.small_size, wrap=14)
    for i in range(6):
        x0, y0 = pos[i]
        x1, y1 = pos[(i + 1) % 6]
        # Shrink toward centers so arrows meet box edges, bowing outward.
        vx, vy = x1 - x0, y1 - y0
        arrow(ax, (x0 + vx * 0.30, y0 + vy * 0.30), (x1 - vx * 0.30, y1 - vy * 0.30),
              curve=-0.25)
    arrow(ax, pos[3], pos[0], curve=-0.3, dashed=True, accent="given")
    label(ax, 56, 44.5, "evaluation sends\nyou back for data", size=STYLE.tiny_size,
          color=STYLE.given)
    arrow(ax, pos[4], pos[2], curve=0.25, dashed=True, accent="given")
    label(ax, 38, 26.0, "deployment reroutes\nto training", size=STYLE.tiny_size,
          color=STYLE.given)
    return save(fig, OUT / "fig_1_13_lifecycle_circle.png")


def fig_1_14_stage_io() -> Path:
    """Each stage consumes the previous stage's output."""
    fig, ax = canvas(118, 34)
    chain = (
        ("collect", "transitions"),
        ("prepare", "prepared data"),
        ("train", "model"),
        ("evaluate", "report"),
        ("deploy", "predictions in use"),
        ("improve", "failures, new data"),
    )
    xs = [11 + i * 19.2 for i in range(6)]
    for x, (name, out) in zip(xs, chain):
        box(ax, x, 20, 15.5, 9, name, accent="learned" if name == "train" else "neutral",
            size=STYLE.small_size)
        label(ax, x + 9.6, 23.2, out, size=STYLE.tiny_size, color=STYLE.ink_soft)
    for x in xs[:-1]:
        arrow(ax, (x + 7.8, 20), (x + 11.4, 20))
    arrow(ax, (xs[-1], 15.5), (xs[0], 15.5), curve=0.22, dashed=True, accent="given",
          label="the chain closes: failures become the next round's data",
          label_offset=-5.6)
    label(ax, 6.2, 26.8, "environment", size=STYLE.tiny_size, color=STYLE.ink_soft)
    arrow(ax, (6.2, 25.4), (8.4, 23.0))
    return save(fig, OUT / "fig_1_14_stage_io.png")


def fig_1_15_improvement_loop() -> Path:
    """The continuous-improvement loop on its own."""
    fig, ax = canvas(96, 52)
    nodes = (
        (48, 43, "deploy", "neutral"),
        (78, 26, "observe failures", "given"),
        (48, 9, "collect targeted data", "given"),
        (18, 26, "update and re-evaluate", "learned"),
    )
    for x, y, name, acc in nodes:
        box(ax, x, y, 26, 9, name, accent=acc, size=STYLE.small_size, wrap=16)
    pts = [(x, y) for x, y, _, _ in nodes]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        vx, vy = b[0] - a[0], b[1] - a[1]
        arrow(ax, (a[0] + vx * 0.28, a[1] + vy * 0.28),
              (b[0] - vx * 0.28, b[1] - vy * 0.28), curve=-0.2)
    label(ax, 18, 15.6, "recorded, comparable runs are what show whether the update helped",
          size=STYLE.tiny_size, color=STYLE.ink_soft, wrap=30)
    return save(fig, OUT / "fig_1_15_improvement_loop.png")





def fig_1_16_roadmap() -> Path:
    """The book's build order after the foundations."""
    fig, ax = canvas(124, 40)
    phases = (
        "run",
        "evaluate",
        "build",
        "train",
        "adapt",
        "improve and\ncompress",
    )
    chapters = ("Ch. 2", "Ch. 3", "Ch. 4-5", "Ch. 6", "Ch. 7-8", "Ch. 9-10")
    xs = [11 + i * 20.5 for i in range(6)]
    for x, name in zip(xs, phases):
        box(ax, x, 22, 17, 11, name, accent="learned", wrap=12,
            size=STYLE.small_size)
    for x in xs[:-1]:
        arrow(ax, (x + 8.5, 22), (x + 12, 22))
    for x, chapter in zip(xs, chapters):
        label(ax, x, 12.7, chapter, size=STYLE.tiny_size, color=STYLE.ink_soft)
    label(ax, 62, 35.0, "from a pretrained model to an efficient one",
          size=STYLE.small_size, color=STYLE.ink_soft)
    return save(fig, OUT / "fig_1_16_roadmap.png")


def fig_1_17_repository() -> Path:
    """The repository by role, with the reusable package at the center."""
    fig, ax = canvas(108, 62)
    box(ax, 54, 33, 30, 13, "src/world_models", sub="the reusable package",
        accent="learned", wrap=18)
    ring = (
        (16, 52, "chapters/", "the text", "neutral"),
        (92, 52, "tests/", "executable checks", "neutral"),
        (16, 12, "figures/ + interactive/", "diagrams and browser experiments", "neutral"),
        (92, 12, "scripts/ + assets/", "generate and store artifacts", "spread"),
    )
    for x, y, name, sub, acc in ring:
        box(ax, x, y, 26, 10, name, sub=sub, accent=acc, size=STYLE.small_size, wrap=18)
    arrow(ax, (28, 49), (44, 39.5), label="reference code")
    arrow(ax, (80, 49), (65, 39.5), label="check behavior", accent="learned")
    arrow(ax, (23, 17.5), (18, 45.5), label="referenced by the text", label_offset=0.4)
    arrow(ax, (64, 27), (83, 16.5), label="produce artifacts", accent="spread",
          label_offset=-3.8)
    return save(fig, OUT / "fig_1_17_repository.png")


FIGURES = [
    fig_1_1_transition,
    fig_1_2_representation,
    fig_1_3_many_futures,
    fig_1_4_reactive_vs_planning,
    fig_1_5_one_state_many_futures,
    fig_1_6_anticipation_better_decisions,
    fig_1_7_simulator_vs_world_model,
    fig_1_8_complementary,
    fig_1_9_two_failures,
    fig_1_10_blueprint,
    fig_1_11_information_flow,
    fig_1_12_responsibility_table,
    fig_1_13_lifecycle_circle,
    fig_1_14_stage_io,
    fig_1_15_improvement_loop,
    fig_1_16_roadmap,
    fig_1_17_repository,
]


def main() -> None:
    for draw in FIGURES:
        path = draw()
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
