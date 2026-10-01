#!/usr/bin/env python3
"""Reproduce Chapter 4 component checks, checkpoint parity, and GPU rollouts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from world_models.cosmos_generation import CHECKPOINT, REVISION
from world_models.flow import euler_step, flow_matching_loss, make_noisy_latents, replace_conditioned
from world_models.models.cosmos_transformer import ScratchCosmosTransformer


ROOT = Path(__file__).resolve().parents[1]
PROMPT = "an aerial view of a sand mining operation; the machinery keeps moving and the water keeps flowing"
SOURCE_FILES = (
    "src/world_models/models/cosmos_transformer.py", "src/world_models/flow.py",
    "src/world_models/cosmos_generation.py", "scripts/chapter_04_experiments.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata() -> dict:
    files = {p: sha256(ROOT / p) for p in SOURCE_FILES}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": CHECKPOINT, "checkpoint_revision": REVISION,
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("torch", "diffusers", "accelerate", "numpy")},
        "source_sha256": files,
        "implementation_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def write_report(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2, allow_nan=False), flush=True)


def require_unused_generation_output(output: Path) -> None:
    """Keep a failed rerun from mixing new latent steps with an older video."""
    if not output.exists():
        return
    artifacts = [output / name for name in ("rollout.mp4", "frames.npy", "manifest.json", "generation.json")]
    artifacts.extend(output.glob("latents_step_*.pt"))
    existing = sorted(path.name for path in artifacts if path.exists())
    if existing:
        raise ValueError(
            f"Generation artifacts already exist in {output}: {', '.join(existing)}. "
            "Choose a new output directory for this run."
        )


def tiny_config() -> dict:
    return dict(num_attention_heads=2, attention_head_dim=16, in_channels=5,
                out_channels=4, num_layers=2, mlp_ratio=2., text_embed_dim=16,
                adaln_lora_dim=8, max_size=(8, 8, 8), patch_size=(1, 2, 2),
                rope_scale=(1., 1., 1.), concat_padding_mask=True,
                extra_pos_embed_type=None, use_crossattn_projection=True,
                crossattn_proj_in_channels=24, encoder_hidden_states_channels=16)


def smoke(args) -> None:
    torch.manual_seed(7)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = ScratchCosmosTransformer.from_config(tiny_config()).to(device)
    clean = torch.randn(1, 4, 3, 4, 6, device=device)
    noise = torch.randn_like(clean)
    mask = torch.zeros(1, 1, 3, 4, 6, device=device)
    mask[:, :, :1] = 1
    t = torch.tensor(0.6, device=device)
    noisy, target = make_noisy_latents(clean, noise, t)
    model_input = replace_conditioned(noisy, clean, mask)
    timestep = mask[:, :, :, :1, :1] * 0.0001 + (1 - mask[:, :, :, :1, :1]) * t
    text = torch.randn(1, 5, 24, device=device)
    padding = torch.zeros(1, 1, 32, 48, device=device)
    prediction = model(hidden_states=model_input, timestep=timestep,
                       encoder_hidden_states=text, condition_mask=mask,
                       padding_mask=padding, return_dict=False)[0]
    loss = flow_matching_loss(prediction, target, mask)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError("Nonfinite gradient")
    exact = euler_step(noise, noise - clean, 1., 0.)
    record = metadata() | {
        "experiment": "random_initialization_component_check", "device": str(device),
        "config": tiny_config(), "parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": list(model_input.shape), "output_shape": list(prediction.shape),
        "loss": float(loss), "gradient_tensors": len(gradients),
        "gradient_norm": float(torch.sqrt(sum(g.float().square().sum() for g in gradients))),
        "oracle_euler_max_error": float((exact - clean).abs().max()),
        "observed_input_max_error": float(((model_input-clean)*mask).abs().max()),
        "scope": "Shape, objective and gradient check with random weights; not a trained video model.",
    }
    write_report(args.output / "smoke.json", record)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = actual.float() - expected.float()
    return {"max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt()),
            "reference_rms": float(expected.float().square().mean().sqrt())}


def tokenizer(args) -> None:
    """Inspect the frozen tokenizer on the exact five-frame conditioning clip."""
    from diffusers import AutoencoderKLWan
    from world_models.observation import load_video
    import imageio.v3 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    observation = load_video(ROOT / "assets/chapter_02/sand_mining.mp4").last(5)
    original = torch.from_numpy(observation.frames.copy()).permute(3, 0, 1, 2).unsqueeze(0)
    video = original.to(device=device, dtype=dtype) / 127.5 - 1
    vae = AutoencoderKLWan.from_pretrained(CHECKPOINT, subfolder="vae", revision=REVISION,
                                         torch_dtype=dtype).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        latents = vae.encode(video).latent_dist.mode()
        decoded = vae.decode(latents).sample
    reconstructed = ((decoded.float().clamp(-1, 1) + 1)*127.5).round().byte()
    pixels = reconstructed[0].permute(1, 2, 3, 0).cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output / "reconstruction.mp4", pixels, fps=16)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4), constrained_layout=True)
    for col, frame in enumerate((0, 2, 4)):
        axes[0, col].imshow(observation.frames[frame])
        axes[0, col].set_title(f"Input frame {frame}")
        axes[1, col].imshow(pixels[frame])
        axes[1, col].set_title(f"VAE reconstruction {frame}")
        for row in range(2):
            axes[row, col].axis("off")
    fig.savefig(args.output / "tokenizer_reconstruction.png", dpi=140)
    plt.close(fig)
    record = metadata() | {"experiment": "conditioning_tokenizer_reconstruction",
        "device": str(device), "dtype": str(dtype), "input_shape": list(video.shape),
        "latent_shape": list(latents.shape), "decoded_shape": list(decoded.shape),
        "reconstruction": error_metrics(decoded.float(), video.float()),
        "seconds": time.perf_counter()-started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    write_report(args.output / "tokenizer.json", record)


def verify(args) -> None:
    from diffusers import CosmosTransformer3DModel

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    torch.manual_seed(19)
    started = time.perf_counter()
    reference = CosmosTransformer3DModel.from_pretrained(
        CHECKPOINT, subfolder="transformer", revision=REVISION, torch_dtype=dtype,
    ).to(device).eval()
    scratch = ScratchCosmosTransformer.from_reference(reference).eval()
    if set(reference.state_dict()) != set(scratch.state_dict()):
        raise AssertionError("Checkpoint keys differ")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Small spatial volume exercises all learned weights without a full video allocation.
    volume = torch.randn(1, 16, 3, 4, 6, device=device, dtype=dtype)
    text = torch.randn(1, 8, 100352, device=device, dtype=dtype)
    mask = torch.zeros(1, 1, 3, 4, 6, device=device, dtype=dtype)
    mask[:, :, :1] = 1
    padding = torch.zeros(1, 1, 32, 48, device=device, dtype=dtype)
    results = []
    tolerance = dict(rtol=0.02, atol=0.02) if dtype == torch.bfloat16 else dict(rtol=2e-4, atol=2e-4)
    with torch.inference_mode():
        for noise_level in (0.05, 0.5, 0.95):
            observed = {}
            handles = []
            for label, network in (("reference", reference), ("scratch", scratch)):
                for i in (0, len(network.transformer_blocks)-1):
                    def capture(_module, _args, value, key=f"{label}/{i}"):
                        observed[key] = value.detach().clone()
                    handles.append(network.transformer_blocks[i].register_forward_hook(capture))
            levels = torch.full((1, 1, 3, 1, 1), noise_level, device=device)
            levels[:, :, :1] = 0.0001
            kwargs = dict(hidden_states=volume, timestep=levels, encoder_hidden_states=text,
                          condition_mask=mask, padding_mask=padding, return_dict=False)
            expected = reference(**kwargs)[0]
            actual = scratch(**kwargs)[0]
            for h in handles:
                h.remove()
            torch.testing.assert_close(actual, expected, **tolerance)
            blocks = {}
            for i in (0, len(reference.transformer_blocks)-1):
                torch.testing.assert_close(observed[f"scratch/{i}"], observed[f"reference/{i}"], **tolerance)
                blocks[str(i)] = error_metrics(observed[f"scratch/{i}"], observed[f"reference/{i}"])
            results.append({"noise_level": noise_level, "velocity": error_metrics(actual, expected), "blocks": blocks})
    if device.type == "cuda":
        torch.cuda.synchronize()
    source = Path(inspect.getfile(CosmosTransformer3DModel))
    record = metadata() | {
        "experiment": "released_checkpoint_numerical_parity", "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "dtype": str(dtype), "parameters": sum(p.numel() for p in scratch.parameters()),
        "state_dict_keys": len(reference.state_dict()), "config": dict(reference.config),
        "reference_source_sha256": sha256(source), "tolerance": tolerance,
        "results": results, "passed": True, "seconds": time.perf_counter()-started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "scope": "All checkpoint weights, a small latent volume, synthetic text features, three noise levels. Video generation is a separate experiment.",
    }
    write_report(args.output / "verify.json", record)


def generate(args) -> None:
    import imageio.v3 as iio
    from world_models.cosmos_generation import generate_video, load_pipeline
    from world_models.experiment import RunManifest
    from world_models.observation import load_video

    require_unused_generation_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    observation = load_video(ROOT / "assets/chapter_02/sand_mining.mp4").last(5)
    record = metadata()
    pipeline = load_pipeline(implementation=args.implementation)
    torch.cuda.reset_peak_memory_stats()
    steps = []
    def capture(index, timestep, latents):
        steps.append({"step": index, "timestep": timestep,
                      "mean": float(latents.mean()), "std": float(latents.std())})
        # Keep a sparse trajectory for numerical comparisons; all these are ignored runtime artifacts.
        if index in {0, args.steps//2, args.steps-1}:
            torch.save(latents.cpu(), args.output / f"latents_step_{index:02d}.pt")
        print(f"step {index+1}/{args.steps}: timestep={timestep:.3f}", flush=True)
    frames = generate_video(
        pipeline, observation, prompt=PROMPT, implementation=args.implementation,
        seed=args.seed, num_frames=args.frames, num_steps=args.steps,
        guidance_scale=args.guidance, offload=args.offload,
        text_cache=args.output.parent / "text_embeddings.pt", callback=capture,
    )
    torch.cuda.synchronize()
    video_path = args.output / "rollout.mp4"
    iio.imwrite(video_path, frames, fps=16)
    # Lossless decoded samples allow a comparison without codec effects.
    np.save(args.output / "frames.npy", frames)
    raw_input_sha = hashlib.sha256(json.dumps(list(observation.frames.shape)).encode()
                                  + observation.frames.tobytes(order="C")).hexdigest()
    manifest = RunManifest(
        checkpoint=CHECKPOINT, revision=REVISION, seed=args.seed, num_frames=args.frames,
        guidance_scale=args.guidance, num_inference_steps=args.steps, height=704, width=1280,
        prompt=PROMPT, observation="assets/chapter_02/sand_mining.mp4", outputs=[str(video_path)],
        conditioning_frames_sha256=raw_input_sha, conditioning_num_frames=5,
        conditioning_fps=observation.fps, output_fps=16,
        implementation=f"chapter04/{args.implementation}",
        implementation_sha256=record["implementation_sha256"],
    )
    manifest.save(args.output)
    record.update({"experiment": "sand_mining_generation", "implementation": args.implementation,
                   "run_id": manifest.run_id, "gpu": torch.cuda.get_device_name(),
                   "seed": args.seed, "guidance": args.guidance, "steps": args.steps,
                   "frames": list(frames.shape), "fps": 16, "offload": args.offload,
                   "scheduler": dict(pipeline.scheduler.config),
                   "seconds": time.perf_counter()-started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "video_sha256": sha256(video_path), "trajectory": steps})
    write_report(args.output / "generation.json", record)


def compare(args) -> None:
    from world_models.experiment import RunManifest, REPRODUCIBILITY_FIELDS, OPTIONAL_REPRODUCIBILITY_FIELDS
    if args.reference.resolve() == args.scratch.resolve():
        raise ValueError("Reference and scratch results must come from different directories")
    reference_manifest = RunManifest.load(args.reference / "manifest.json")
    scratch_manifest = RunManifest.load(args.scratch / "manifest.json")
    for expected, manifest in (("reference", reference_manifest), ("scratch", scratch_manifest)):
        if manifest.implementation != f"chapter04/{expected}":
            raise ValueError(f"Expected a chapter04/{expected} implementation in its comparison directory")
    for field in (*REPRODUCIBILITY_FIELDS, *OPTIONAL_REPRODUCIBILITY_FIELDS):
        if field not in {"implementation", "implementation_sha256"}:
            if getattr(reference_manifest, field) != getattr(scratch_manifest, field):
                raise ValueError(f"Generation configurations differ in {field}")
    reference_generation = json.loads((args.reference / "generation.json").read_text())
    scratch_generation = json.loads((args.scratch / "generation.json").read_text())
    for expected, manifest, generation in (
        ("reference", reference_manifest, reference_generation),
        ("scratch", scratch_manifest, scratch_generation),
    ):
        if generation.get("run_id") != manifest.run_id or generation.get("implementation") != expected:
            raise ValueError(f"The {expected} generation report does not match its manifest")
        if not isinstance(generation.get("scheduler"), dict) or not generation["scheduler"]:
            raise ValueError(f"The {expected} generation report needs a scheduler configuration")
    if reference_generation["scheduler"] != scratch_generation["scheduler"]:
        raise ValueError("Scheduler configurations differ between reference and scratch runs")
    reference_steps = {p.name for p in args.reference.glob("latents_step_*.pt")}
    scratch_steps = {p.name for p in args.scratch.glob("latents_step_*.pt")}
    if not reference_steps or reference_steps != scratch_steps:
        raise ValueError("Expected identical nonempty sets of saved latent steps")
    first, second = np.load(args.reference / "frames.npy"), np.load(args.scratch / "frames.npy")
    if first.shape != second.shape:
        raise ValueError("Video shapes differ")
    delta = first.astype(np.float32) - second.astype(np.float32)
    latent_results = {}
    for path in sorted(args.reference.glob("latents_step_*.pt")):
        a = torch.load(path, map_location="cpu", weights_only=True)
        b = torch.load(args.scratch / path.name, map_location="cpu", weights_only=True)
        latent_results[path.name] = error_metrics(b, a)
    record = metadata() | {"experiment": "paired_generation_comparison",
        "reference_run_id": reference_manifest.run_id,
        "scratch_run_id": scratch_manifest.run_id,
        "reference_generation_sha256": sha256(args.reference / "generation.json"),
        "scratch_generation_sha256": sha256(args.scratch / "generation.json"),
        "scheduler": reference_generation["scheduler"],
        "frames": list(first.shape), "decoded_uint8_max_abs": float(np.abs(delta).max()),
        "decoded_uint8_mae": float(np.abs(delta).mean()), "decoded_uint8_rmse": float(np.sqrt((delta**2).mean())),
        "latent_comparison": latent_results}
    write_report(args.output / "comparison.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "verify", "tokenizer", "generate", "compare"):
        sub = subs.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
        if name in {"smoke", "verify", "tokenizer"}:
            sub.add_argument("--device", default="cpu" if name == "smoke" else "cuda")
        if name == "generate":
            sub.add_argument("--implementation", choices=("reference", "scratch"), default="scratch")
            sub.add_argument("--frames", type=int, default=29)
            sub.add_argument("--steps", type=int, default=15)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--guidance", type=float, default=7.)
            sub.add_argument("--offload", action="store_true")
        if name == "compare":
            sub.add_argument("--reference", type=Path, required=True)
            sub.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {"smoke": smoke, "verify": verify, "tokenizer": tokenizer,
     "generate": generate, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
