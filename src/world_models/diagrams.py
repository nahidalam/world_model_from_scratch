"""Primitives for the book's schematic figures.

Every figure in the book is a diagram rather than a data plot: boxes, arrows,
and labels that teach one idea. This module supplies the shared vocabulary so
that a reader who has understood one figure can read the next without relearning
its conventions.

Colour is used semantically and never alone. Three accents carry meaning:

    LEARNED  (blue)   anything the model acquires from data
    GIVEN    (orange) anything a person wrote down or the world supplied
    SPREAD   (aqua)   uncertainty, several possible futures

Every accent-bearing element also carries a text label, so the figures survive
grayscale printing and colour-vision deficiency. The palette validates on the
data-visualisation skill's all-pairs check in both light and dark modes.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

Accent = Literal["learned", "given", "spread", "neutral"]


@dataclass(frozen=True)
class Style:
    """Colours and sizes shared by every figure."""

    surface: str = "#fcfcfb"
    ink: str = "#0b0b0b"
    ink_soft: str = "#52514e"
    ink_faint: str = "#8f8e88"
    rule: str = "#d6d5cf"

    learned: str = "#2a78d6"
    given: str = "#eb6834"
    spread: str = "#1baf7a"

    learned_fill: str = "#e6f0fc"
    given_fill: str = "#fdece4"
    spread_fill: str = "#e3f6ef"
    neutral_fill: str = "#f2f1ec"

    title_size: float = 12.0
    label_size: float = 10.5
    small_size: float = 9.0
    tiny_size: float = 8.0

    line_width: float = 1.6
    box_radius: float = 0.9

    font_stack: tuple[str, ...] = field(
        default_factory=lambda: ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
    )


STYLE = Style()

_EDGE = {
    "learned": STYLE.learned,
    "given": STYLE.given,
    "spread": STYLE.spread,
    "neutral": STYLE.ink_soft,
}
_FILL = {
    "learned": STYLE.learned_fill,
    "given": STYLE.given_fill,
    "spread": STYLE.spread_fill,
    "neutral": STYLE.neutral_fill,
}


def canvas(width: float = 100.0, height: float = 50.0, scale: float = 0.10):
    """Return a blank figure and axes in a 0..width by 0..height coordinate space."""
    if width <= 0 or height <= 0:
        raise ValueError(f"canvas dimensions must be positive, got {width}x{height}")

    plt.rcParams["font.family"] = list(STYLE.font_stack)
    fig, ax = plt.subplots(figsize=(width * scale, height * scale))
    fig.patch.set_facecolor(STYLE.surface)
    ax.set_facecolor(STYLE.surface)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    sub: str | None = None,
    accent: Accent = "neutral",
    dashed: bool = False,
    faded: bool = False,
    wrap: int = 18,
    size: float | None = None,
) -> tuple[float, float]:
    """Draw a labelled rounded box centred on (x, y). Returns its centre."""
    alpha = 0.45 if faded else 1.0
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={STYLE.box_radius}",
            linewidth=STYLE.line_width,
            edgecolor=_EDGE[accent],
            facecolor=_FILL[accent],
            linestyle="--" if dashed else "-",
            alpha=alpha,
            zorder=2,
        )
    )
    text_y = y + (1.6 if sub else 0.0)
    ax.text(
        x,
        text_y,
        textwrap.fill(label, wrap),
        ha="center",
        va="center",
        fontsize=size or STYLE.label_size,
        color=STYLE.ink,
        alpha=alpha,
        zorder=3,
    )
    if sub:
        ax.text(
            x,
            y - 2.6,
            textwrap.fill(sub, wrap + 8),
            ha="center",
            va="center",
            fontsize=STYLE.tiny_size,
            color=STYLE.ink_soft,
            alpha=alpha,
            zorder=3,
        )
    return x, y


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    accent: Accent = "neutral",
    curve: float = 0.0,
    dashed: bool = False,
    faded: bool = False,
    label_offset: float = 1.6,
    size: float | None = None,
) -> None:
    """Draw an arrow, optionally curved (``curve`` bows it left of travel)."""
    alpha = 0.4 if faded else 1.0
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={curve}",
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=STYLE.line_width,
            color=_EDGE[accent],
            linestyle="--" if dashed else "-",
            alpha=alpha,
            zorder=1,
        )
    )
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + label_offset + abs(curve) * 8,
            label,
            ha="center",
            va="bottom",
            fontsize=size or STYLE.tiny_size,
            color=STYLE.ink_soft,
            alpha=alpha,
            zorder=3,
        )


def label(
    ax,
    x: float,
    y: float,
    s: str,
    *,
    size: float | None = None,
    color: str | None = None,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    wrap: int | None = None,
    style: str = "normal",
) -> None:
    """Place free-standing text."""
    ax.text(
        x,
        y,
        textwrap.fill(s, wrap) if wrap else s,
        ha=ha,
        va=va,
        fontsize=size or STYLE.small_size,
        color=color or STYLE.ink,
        fontweight=weight,
        fontstyle=style,
        zorder=4,
    )


def panel(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    accent: Accent = "neutral",
) -> None:
    """Draw a titled background region used to separate two halves of a figure."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.2",
            linewidth=1.0,
            edgecolor=STYLE.rule,
            facecolor="none",
            zorder=0,
        )
    )
    ax.text(
        x - w / 2 + 2,
        y + h / 2 - 2.4,
        title,
        ha="left",
        va="center",
        fontsize=STYLE.title_size,
        color=_EDGE[accent],
        fontweight="bold",
        zorder=4,
    )


def dot(ax, x: float, y: float, r: float = 0.9, *, accent: Accent = "neutral", faded: bool = False) -> None:
    """Draw a small filled node."""
    ax.add_patch(
        Circle(
            (x, y),
            r,
            linewidth=STYLE.line_width,
            edgecolor=_EDGE[accent],
            facecolor=_FILL[accent],
            alpha=0.45 if faded else 1.0,
            zorder=3,
        )
    )


def wedge(ax, points: list[tuple[float, float]], *, accent: Accent = "spread", alpha: float = 0.18) -> None:
    """Fill a polygon, used for uncertainty cones and coverage regions."""
    ax.add_patch(
        Polygon(points, closed=True, facecolor=_EDGE[accent], edgecolor="none", alpha=alpha, zorder=0)
    )


def save(fig, path: Path | str, *, dpi: int = 200) -> Path:
    """Write the figure to ``path``, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=STYLE.surface, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return out
