"""Build Chapter 4's vector diagrams using only the Python standard library."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "chapter_04"
INK, SOFT, BLUE, ORANGE = "#14243b", "#42536a", "#2a78d6", "#d85c2c"


class Figure:
    def __init__(self, title: str, description: str, height: int, width: int = 1120, show_title: bool = True):
        self.height = height
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#53637a"/></marker></defs>',
            f'<rect width="{width}" height="{height}" rx="16" fill="#f7f9fc"/>',
            '<g font-family="system-ui, sans-serif">',
        ]
        if show_title:
            self.text(32, 40, title, 24, weight=600)

    def text(self, x: float, y: float, label: str, size: int = 16, color: str = INK, weight: int = 400, anchor: str = "start"):
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(label)}</text>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "#fff", stroke: str = "#c2cedc", rx: int = 10, dashed: bool = False):
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')

    def box(self, x: int, y: int, w: int, h: int, heading: str, lines: tuple[str, ...] = (), fill: str = "#fff", stroke: str = "#c2cedc", top: bool = False):
        # Center the text vertically unless the caller draws cells under a top heading.
        self.rect(x, y, w, h, fill, stroke)
        baseline = y + 28 if top else y + (h - 18 - 23 * len(lines)) / 2 + 15
        self.text(x + w / 2, baseline, heading, 17, weight=600, anchor="middle")
        for i, line in enumerate(lines):
            self.text(x + w / 2, baseline + 25 + i * 23, line, 14, SOFT, anchor="middle")

    def arrow(self, points: str):
        self.items.append(f'<path d="{points}" fill="none" stroke="#53637a" stroke-width="2"/>')
        # Explicit arrowheads also render in native SVG viewers without markers.
        x = y = prev_x = prev_y = 0.0
        for command, arguments in re.findall(r"([MHV])([^MHV]+)", points):
            prev_x, prev_y = x, y
            values = [float(value) for value in arguments.split()]
            if command == "M":
                x, y = values
            elif command == "H":
                x = values[0]
            else:
                y = values[0]
        dx, dy = x - prev_x, y - prev_y
        distance = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / distance, dy / distance
        left = (x - 9 * ux + 4 * uy, y - 9 * uy - 4 * ux)
        right = (x - 9 * ux - 4 * uy, y - 9 * uy + 4 * ux)
        self.items.append(f'<polygon points="{x},{y} {left[0]},{left[1]} {right[0]},{right[1]}" fill="#53637a"/>')

    def write(self, name: str):
        (OUT / name).write_text("\n".join(self.items + ["</g></svg>"]) + "\n")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = Figure("Inside Cosmos-Predict2.5-2B", "The observed video passes through the pretrained VAE encoder into a latent with observed and noise frames. A pretrained text encoder turns the prompt into features that the 28-block Transformer reads. The Transformer and sampler repeat until t equals zero, and the pretrained VAE decoder produces the video.", 290, width=1160, show_title=False)
    f.box(20, 50, 150, 80, "Observed video")
    f.arrow("M170 90H193")
    f.box(204, 50, 150, 80, "VAE encoder", ("pretrained",))
    f.arrow("M354 90H377")
    f.box(388, 50, 170, 80, "Latent", ("observed + noise",))
    f.arrow("M558 90H581")
    f.box(592, 50, 170, 80, "Transformer", ("28 blocks",), "#fff", INK)
    f.arrow("M762 90H785")
    f.box(796, 50, 170, 80, "Sampler", ("step toward t = 0",))
    f.arrow("M881 50V30H677V39")
    f.box(388, 190, 170, 80, "Text prompt")
    f.arrow("M558 230H581")
    f.box(592, 190, 170, 80, "Text encoder", ("pretrained",))
    f.arrow("M677 190V141")
    f.arrow("M881 130V179")
    f.box(796, 190, 170, 80, "VAE decoder", ("after the last step",))
    f.arrow("M966 230H989")
    f.box(1000, 190, 140, 80, "Video")
    f.write("cosmos_reference_architecture.svg")

    # Figure 4.1 shows the model's parts, not the sampling loop. All words sit inside blocks;
    # the Markdown caption carries the explanation.
    observed_fill, future_fill = "#e6f0fc", "#fdece4"
    f = Figure("The small world model", "Five observed frames pass through the frozen VAE encoder into two known latent frames; three future latent frames are unknown. The trained Transformer fills in the future latents, and the frozen VAE decoder produces 17 frames: 5 observed and 12 generated.", 140, width=1160, show_title=False)
    f.box(20, 20, 150, 100, "Frames 0–4", top=True)
    for i in range(5):
        f.rect(33 + i * 26, 64, 20, 40, observed_fill, BLUE, rx=3)
    f.arrow("M170 70H191")
    f.box(202, 20, 130, 100, "VAE encoder", ("frozen",))
    f.arrow("M332 70H353")
    f.box(364, 20, 190, 100, "Latents", top=True)
    for i in range(5):
        known = i < 2
        f.rect(372 + i * 36, 62, 30, 44, observed_fill if known else "#fff", BLUE if known else ORANGE, rx=4, dashed=not known)
    f.arrow("M554 70H575")
    f.box(586, 20, 150, 100, "Transformer", ("trained",), "#fff", INK)
    f.arrow("M736 70H757")
    f.box(768, 20, 130, 100, "VAE decoder", ("frozen",))
    f.arrow("M898 70H919")
    f.box(930, 20, 210, 100, "Frames 0–16", top=True)
    for i in range(17):
        observed = i < 5
        f.rect(935 + i * 12, 64, 8, 40, observed_fill if observed else future_fill, BLUE if observed else ORANGE, rx=2)
    f.write("architecture.svg")

    # Every figure below follows the same rules as Figure 4.1: no text outside blocks,
    # and an arrow for every transition. The Markdown caption carries the explanation.

    f = Figure("From frames to tokens", "The VAE encoder compresses 17 RGB frames at 128 by 128 into a latent with 16 channels, 5 frames, and a 16 by 16 grid. Grouping 1 by 2 by 2 latent cells gives 320 patches, and the input projection turns each patch into a 384-feature token.", 140, width=1160, show_title=False)
    f.box(20, 20, 196, 100, "Video frames", ("17 frames", "128 × 128 RGB"))
    f.arrow("M216 70H239")
    f.box(250, 20, 196, 100, "VAE encoder", ("time: 17 → 5", "space: 128 → 16"))
    f.arrow("M446 70H469")
    f.box(480, 20, 196, 100, "Latent", ("16 channels", "5 × 16 × 16"))
    f.arrow("M676 70H699")
    f.box(710, 20, 196, 100, "Patches", ("1 × 2 × 2 cells each", "320 patches"))
    f.arrow("M906 70H929")
    f.box(940, 20, 196, 100, "Tokens", ("input projection", "384 features each"), "#fff", INK)
    f.write("latent_tokens.svg")

    f = Figure("One attention head", "Video tokens are projected into queries, keys, and values. Queries and keys are normalized and rotated by position, compared to give scores, and passed through softmax to give weights. The weights combine the values into output tokens.", 300, width=1160, show_title=False)
    f.box(20, 110, 150, 100, "Video tokens", ("one per patch",))
    f.arrow("M170 160H190V70H199")
    f.arrow("M170 160H190V150H199")
    f.arrow("M170 160H190V250H199")
    f.box(210, 40, 110, 60, "Query")
    f.box(210, 120, 110, 60, "Key")
    f.box(210, 220, 110, 60, "Value")
    f.arrow("M320 70H349")
    f.arrow("M320 150H349")
    f.box(360, 40, 200, 140, "Prepare Q and K", ("RMS normalize", "rotate by position", "(RoPE)"), "#fff", INK)
    f.arrow("M560 110H589")
    f.box(600, 70, 170, 80, "Scores", ("QKᵀ / √d",))
    f.arrow("M770 110H799")
    f.box(810, 70, 160, 80, "Weights", ("softmax per row",))
    f.arrow("M970 110H999")
    f.arrow("M320 250H999")
    f.box(1010, 70, 130, 210, "Output", ("weights × V",))
    f.write("attention.svg")

    f = Figure("Flow matching", "Training mixes a clean latent and noise at level t, and the Transformer's predicted velocity is compared with noise minus the clean latent. Generation starts from noise; the Transformer predicts a velocity, the latent takes a small step toward t equals zero, and the loop repeats until the generated latent remains.", 360, width=1000, show_title=False)
    f.box(20, 20, 190, 70, "Clean latent z", ("t = 0",), "#e6f0fc", BLUE)
    f.box(20, 110, 190, 70, "Noise ε", ("t = 1",), "#fdece4", ORANGE)
    f.arrow("M210 55H225V100H239")
    f.arrow("M210 145H225V100H239")
    f.box(250, 60, 230, 80, "Mix at level t", ("(1 − t)z + tε",))
    f.arrow("M480 100H509")
    f.box(520, 60, 190, 80, "Transformer", ("predicts velocity",), "#fff", INK)
    f.arrow("M710 100H739")
    f.box(750, 60, 230, 80, "Loss", ("prediction vs. ε − z",))
    f.box(20, 230, 190, 80, "Noise ε", ("t = 1",), "#fdece4", ORANGE)
    f.arrow("M210 270H239")
    f.box(250, 230, 230, 80, "Transformer", ("predicts velocity",), "#fff", INK)
    f.arrow("M480 270H509")
    f.box(520, 230, 190, 80, "Update latent", ("small step toward t = 0",))
    f.arrow("M615 310V335H365V321")
    f.arrow("M710 270H739")
    f.box(750, 230, 230, 80, "Generated latent", ("t = 0",), "#e6f0fc", BLUE)
    f.write("flow_matching.svg")

    f = Figure("One Transformer block", "Tokens pass through self-attention, cross-attention, and a feed-forward network. Each sublayer applies adaptive normalization and a gated residual addition. The noise level t becomes a timestep embedding that sets each sublayer's shift, scale, and gate. The learned context enters through cross-attention.", 380, width=1080, show_title=False)
    f.box(204, 20, 200, 70, "Noise level t")
    f.arrow("M404 55H427")
    f.box(438, 20, 200, 70, "Timestep embedding")
    f.arrow("M638 55H661")
    f.box(672, 20, 200, 70, "Shift, scale, gate", ("one set per sublayer",))
    f.arrow("M772 90V115H304V129")
    f.arrow("M772 90V115H538V129")
    f.arrow("M772 90V129")
    f.box(20, 160, 150, 80, "Tokens in")
    f.arrow("M170 200H193")
    f.box(204, 140, 200, 120, "Self-attention", ("adaptive norm", "attend to all tokens", "gated residual add"), "#fff", INK)
    f.arrow("M404 200H427")
    f.box(438, 140, 200, 120, "Cross-attention", ("adaptive norm", "attend to context", "gated residual add"), "#fff", INK)
    f.arrow("M638 200H661")
    f.box(672, 140, 200, 120, "Feed-forward", ("adaptive norm", "per-token MLP", "gated residual add"), "#fff", INK)
    f.arrow("M872 200H895")
    f.box(906, 160, 150, 80, "Tokens out", ("to the next block",))
    f.box(438, 300, 200, 60, "Learned context")
    f.arrow("M538 300V271")
    f.write("transformer_block.svg")

    observed_fill, future_fill = "#e6f0fc", "#fdece4"
    f = Figure("Condition on observed frames", "Five observed frames pass through the VAE encoder into two observed latent frames. Three frames to generate start as noise. A mask marks observed positions with 1 and positions to generate with 0, and is concatenated with the latent as a seventeenth channel before the Transformer.", 270, width=1000, show_title=False)
    f.box(20, 20, 150, 100, "Frames 0–4", top=True)
    for i in range(5):
        f.rect(33 + i * 26, 64, 20, 40, observed_fill, BLUE, rx=3)
    f.arrow("M170 70H193")
    f.box(204, 20, 130, 100, "VAE encoder")
    f.arrow("M334 70H357")
    f.box(368, 20, 220, 100, "Latent", top=True)
    for i in range(5):
        known = i < 2
        f.rect(376 + i * 42, 62, 36, 44, observed_fill if known else future_fill, BLUE if known else ORANGE, rx=4)
    f.arrow("M588 70H611")
    f.box(622, 20, 170, 100, "Concatenate", ("16 + 1 channels",))
    f.arrow("M792 70H815")
    f.box(826, 20, 150, 100, "Transformer", ("trained",), "#fff", INK)
    f.box(398, 170, 160, 90, "Noise", top=True)
    for i in range(3):
        f.rect(418 + i * 42, 206, 36, 44, future_fill, ORANGE, rx=4)
    f.arrow("M478 170V131")
    f.box(622, 170, 170, 90, "Mask", top=True)
    for i in range(5):
        known = i < 2
        f.rect(632 + i * 31, 206, 26, 44, observed_fill if known else "#fff", BLUE if known else ORANGE, rx=4)
        f.text(645 + i * 31, 234, "1" if known else "0", 15, BLUE if known else ORANGE, weight=600, anchor="middle")
    f.arrow("M707 170V131")
    f.write("conditioning.svg")


if __name__ == "__main__":
    build()
