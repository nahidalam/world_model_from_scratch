#!/usr/bin/env python3
"""Copy Chapter 4 experiment evidence into the book and draw its comparison figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/chapter_04")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/chapter_04")
    parser.add_argument("--generation-script", type=Path,
                        default=ROOT / "scripts/chapter_04_experiments.py",
                        help="Exact experiment script used for the saved generation runs.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / "results"
    results.mkdir(exist_ok=True)
    for command in ("smoke", "verify", "tokenizer", "comparison"):
        shutil.copy2(args.input / command / f"{command}.json", results / f"{command}.json")
    for implementation in ("reference", "scratch"):
        for name in ("generation", "manifest"):
            shutil.copy2(args.input / implementation / f"{name}.json",
                         results / f"{implementation}_{name}.json")
        shutil.copy2(args.input / implementation / "rollout.mp4",
                     args.output / f"{implementation}_rollout.mp4")
    shutil.copy2(args.input / "tokenizer/tokenizer_reconstruction.png",
                 args.output / "tokenizer_reconstruction.png")
    shutil.copy2(args.input / "tokenizer/reconstruction.mp4",
                 args.output / "tokenizer_reconstruction.mp4")

    # Keep the exact code named by the generation reports, including scripts
    # that may later acquire additional argument or result-validation checks.
    report = json.loads((args.input / "reference/generation.json").read_text())
    scratch_report = json.loads((args.input / "scratch/generation.json").read_text())
    if report["source_sha256"] != scratch_report["source_sha256"]:
        raise ValueError("The two generation runs used different source files")
    with zipfile.ZipFile(results / "generation_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, expected in report["source_sha256"].items():
            path = args.generation_script if relative == "scripts/chapter_04_experiments.py" else ROOT / relative
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source snapshot does not match generation report: {relative}")
            archive.writestr(relative, data)

    reference = np.load(args.input / "reference/frames.npy", mmap_mode="r")
    scratch = np.load(args.input / "scratch/frames.npy", mmap_mode="r")
    if reference.shape != scratch.shape:
        raise ValueError("The two generated videos have different shapes")
    selected = (4, 12, 20, len(reference)-1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 5.5), constrained_layout=True)
    for row, (label, frames) in enumerate((("Reference", reference), ("Our implementation", scratch))):
        for column, index in enumerate(selected):
            axes[row, column].imshow(frames[index])
            axes[row, column].set_title(f"{label} · frame {index}", fontsize=12)
            axes[row, column].axis("off")
    fig.savefig(args.output / "rollout_comparison.png", dpi=140)
    plt.close(fig)
    print(f"Wrote experiment reports, videos, source snapshot, and figure to {args.output}")


if __name__ == "__main__":
    main()
