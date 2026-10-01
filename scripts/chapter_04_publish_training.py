#!/usr/bin/env python3
"""Publish checked Chapter 4 training reports, figures, examples, and source files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from world_models.experiment import RunManifest


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_sources(reports: dict[str, dict]) -> dict[str, bytes]:
    """Resolve each recorded source name and require the recorded exact bytes."""
    sources: dict[str, bytes] = {}
    for label, report in reports.items():
        field = "source_sha256" if label == "cache" else "source"
        for name, expected in report[field].items():
            relative = Path(name)
            path = (ROOT / relative if relative.parts[0] in ("scripts", "src")
                    else ROOT / "src/world_models" / relative).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError(f"Source does not match {label} report: {path}")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Source does not match {label} report: {path}")
            sources[str(path.relative_to(ROOT))] = content
    return sources


def selected_indices(rows: list[dict]) -> list[tuple[int, list[str]]]:
    ranked = sorted(range(len(rows)), key=lambda index: (
        rows[index]["generated"]["centroid_error_penalized_px"], rows[index]["trajectory_seed"]))
    selections: dict[int, list[str]] = {}
    for index, label in ((0, "fixed first example"), (ranked[0], "best error"),
                         (ranked[len(ranked) // 2], "median error"), (ranked[-1], "worst error")):
        selections.setdefault(index, []).append(label)
    return list(selections.items())


def checked_example(directory: Path, report: dict, index: int, roles: list[str], group: str) -> dict:
    """Keep the original run ID and verify its checkpoint and implementation scope."""
    row = report["trajectories"][index]
    source = directory / f"trajectory_{row['trajectory_seed']}"
    manifest = RunManifest.load(source / "manifest.json")
    inference = read_json(source / "inference.json")
    implementation_hash = hashlib.sha256(json.dumps(inference, sort_keys=True).encode()).hexdigest()
    observation = f"{report['cache_id']}/{report['split']}/{row['trajectory_seed']}"
    if (manifest.run_id != row["run_id"] or manifest.revision != report["checkpoint_sha256"]
            or manifest.observation != observation or manifest.implementation_sha256 != implementation_hash
            or inference != report["inference"] or read_json(source / "metrics.json") != row):
        raise ValueError(f"Example identity or implementation scope differs from its evaluation: {source}")
    filenames = tuple(dict.fromkeys(("comparison.png", "rollout.mp4", "ground_truth.mp4", "persistence.mp4",
                                    "manifest.json", "metrics.json", "inference.json", *manifest.outputs)))
    for name in filenames:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Manifest outputs must stay within their example directory: {source / name}")
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    return {"index": index, "roles": roles, "trajectory_seed": row["trajectory_seed"],
            "run_id": manifest.run_id, "centroid_error_penalized_px": row["generated"]["centroid_error_penalized_px"],
            "source": source, "destination": Path("examples") / group / source.name, "files": filenames}


def loss_figure(rows: list[dict], path: Path) -> None:
    steps = np.asarray([row["step"] for row in rows])
    losses = np.asarray([row["loss"] for row in rows])
    window = min(25, len(rows))
    ends = np.arange(1, len(rows) + 1)
    starts = np.maximum(0, ends - window)
    cumulative = np.concatenate(([0.0], np.cumsum(losses)))
    smooth = (cumulative[ends] - cumulative[starts]) / (ends - starts)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(steps, losses, color="#3278ad", alpha=0.23, linewidth=0.8, label="Training loss · raw")
    ax.plot(steps, smooth, color="#216493", linewidth=2, label=f"Training loss · trailing {window} updates")
    validation = [row for row in rows if "validation_loss" in row]
    if validation:
        ax.plot([row["step"] for row in validation], [row["validation_loss"] for row in validation],
                color="#bf5a25", marker="o", markersize=4, linewidth=1.6, label="Validation loss · fixed noise")
    ax.set_yscale("log")
    ax.set(title="Future-region flow-matching loss", xlabel="Optimizer update",
           ylabel="Mean squared velocity error (log scale)")
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def motion_figure(report: dict, cache: dict, path: Path) -> None:
    rows = report["trajectories"]
    generated = np.asarray([row["generated"]["centroid_error_penalized_px"] for row in rows])
    persistence = np.asarray([row["persistence"]["centroid_error_penalized_px"] for row in rows])
    x = np.arange(1, len(rows) + 1)
    diagonal = math.sqrt(2) * cache["spec"]["size"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.24)
    target = report.get("tracking", {}).get("target", "object")
    fig.suptitle(f"Future {target} motion on {len(rows)} {report['split']} trajectories", x=0.09, ha="left", y=0.98)
    ax.vlines(x, np.minimum(generated, persistence), np.maximum(generated, persistence), color="#cbd4dc", linewidth=1)
    ax.scatter(x, persistence, color="#bf5a25", marker="x", s=25, label="Repeat last observed frame", zorder=3)
    ax.scatter(x, generated, color="#216493", s=23, label="Generated future", zorder=4)
    ax.set(xlabel="Trajectory in saved evaluation order", ylabel=f"Mean future {target} center error (pixels)", ylim=(0, None))
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    coverage = report["aggregate"]["generated"]["object_detection_fraction"]
    fig.text(0.09, 0.09, f"Each pair uses the same trajectory and future frames. Missing detections receive {diagonal:.1f} px per frame.\n"
             f"Generated {target} detection coverage: {coverage:.1%}. Lines connect the two methods for each trajectory.",
             fontsize=9, color="#43515d", linespacing=1.6)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def publish_training(*, training: Path, evaluation: Path, overfit_training: Path, overfit_evaluation: Path,
                     cache: Path, reconstruction: Path, resume: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh publication output directory.")
    inputs = {
        "training": training / "training.json", "evaluation": evaluation / "evaluation.json",
        "overfit_training": overfit_training / "training.json",
        "overfit_evaluation": overfit_evaluation / "evaluation.json", "cache": cache / "cache.json",
        "reconstruction_validation": reconstruction / "reconstruction_validation.json",
        "resume_verification": resume / "resume_verification.json",
    }
    reports = {name: read_json(path) for name, path in inputs.items()}
    for prefix in ("", "overfit_"):
        trained, evaluated = reports[prefix + "training"], reports[prefix + "evaluation"]
        for field in ("cache_id", "checkpoint_sha256"):
            if trained[field] != evaluated[field]:
                raise ValueError(f"{prefix or 'main ' }training/evaluation {field} differs.")
        if not evaluated["trajectories"] or evaluated["examples"] != len(evaluated["trajectories"]):
            raise ValueError(f"{prefix or 'main '}evaluation needs one trajectory record per example.")
        if (evaluated["inference"]["cache_id"] != evaluated["cache_id"]
                or evaluated["inference"]["source"] != evaluated["source"]):
            raise ValueError(f"{prefix or 'main '}evaluation inference scope differs from its report.")
    main_cache_id = reports["training"]["cache_id"]
    if main_cache_id != reports["cache"]["cache_id"]:
        raise ValueError("Training and cache.json must identify the same latent cache.")
    sources = checked_sources(reports)
    rows = [json.loads(line) for line in (training / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not rows or not all(math.isfinite(row["loss"]) for row in rows):
        raise ValueError("Training metrics must include finite losses for the loss figure.")
    examples = [checked_example(evaluation, reports["evaluation"], index, roles, "main")
                for index, roles in selected_indices(reports["evaluation"]["trajectories"])]
    examples.append(checked_example(overfit_evaluation, reports["overfit_evaluation"], 0,
                                    ["fixed first overfit example"], "overfit"))
    reconstruction_pairs = []
    for row in reports["reconstruction_validation"]["trajectories"]:
        relative = Path(row["directory"])
        filenames = {kind: row["reconstruction"][kind] for kind in ("png", "video")}
        for name in filenames.values():
            entry = relative / name
            if entry.is_absolute() or ".." in entry.parts:
                raise ValueError(f"Reconstruction media must stay within its input directory: {entry}")
            if not (reconstruction / entry).is_file():
                raise FileNotFoundError(reconstruction / entry)
        reconstruction_pairs.append({"trajectory_seed": row["trajectory_seed"],
                                     "source": reconstruction / relative,
                                     "destination": Path("reconstruction") / relative,
                                     "files": filenames, "layout": row["reconstruction"]["layout"]})
    output.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        shutil.copy2(path, output / f"{name}.json")
    shutil.copy2(training / "metrics.jsonl", output / "metrics.jsonl")
    optional = ((training / "training_initial.json", "training_initial.json"),
                (overfit_training / "training_initial.json", "overfit_training_initial.json"),
                (overfit_training / "metrics.jsonl", "overfit_metrics.jsonl"),
                (resume / "resumed/training_initial.json", "resume_training_initial.json"))
    for path, name in optional:
        if path.is_file():
            shutil.copy2(path, output / name)
    loss_figure(rows, output / "loss_curve.png")
    motion_figure(reports["evaluation"], reports["cache"], output / "motion_error.png")
    for example in examples:
        destination = output / example["destination"]
        destination.mkdir(parents=True)
        for name in example["files"]:
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example["source"] / name, destination / name)
    for pair in reconstruction_pairs:
        for name in pair["files"].values():
            destination = output / pair["destination"] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pair["source"] / name, destination)
    with zipfile.ZipFile(output / "training_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, content in sorted(sources.items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    publication = {
        "schema_version": 1,
        "experiment": "chapter04_training_publication",
        "generator": reports["cache"]["generator"],
        "tracking": reports["evaluation"].get("tracking"),
        "inputs": {name: str(path.resolve()) for name, path in inputs.items()},
        "cache_roles": {name: {"cache_id": reports[name]["cache_id"],
                               "matches_main_training_cache": reports[name]["cache_id"] == main_cache_id}
                        for name in ("training", "overfit_training", "reconstruction_validation", "resume_verification")},
        "selection_criterion": "Main examples are fixed index 0 plus the lowest, upper-median, and highest "
                               "generated penalized future-center errors; ties use trajectory seed. Repeated "
                               "selections share one artifact directory. Overfit uses fixed index 0.",
        "examples": [{key: str(value) if isinstance(value, Path) else value for key, value in example.items()}
                     for example in examples],
        "reconstruction_selection": "All trajectories from the reconstruction report, in saved order.",
        "reconstruction_pairs": [
            {key: str(value) if isinstance(value, Path) else value for key, value in pair.items()}
            for pair in reconstruction_pairs],
        "source_archive": "training_source.zip",
        "source_sha256": {relative: hashlib.sha256(content).hexdigest() for relative, content in sorted(sources.items())},
        "publisher_sha256": sha256(Path(__file__)),
        "artifact_sha256": {str(path.relative_to(output)): sha256(path)
                            for path in sorted(output.rglob("*")) if path.is_file()},
        "scope": "Reports and run manifests retain their original identities and implementation hashes. "
                 "Reconstruction and resume reports identify their own caches, including separate probe "
                 "datasets. Motion errors include the image-diagonal penalty for missing detections.",
    }
    (output / "publication.json").write_text(json.dumps(publication, indent=2, allow_nan=False) + "\n")
    return publication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "evaluation", "overfit-training", "overfit-evaluation", "cache",
                 "reconstruction", "resume", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = publish_training(**vars(args))
    print(f"Published {len(report['examples'])} example bundles and checked training evidence to {args.output}")


if __name__ == "__main__":
    main()
