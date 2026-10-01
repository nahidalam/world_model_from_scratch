"""Chapter 4 generation with the book's Transformer and visible sampling loop.

Frozen text/video encoders and the safety checks come from the Chapter 2
pipeline. This module owns the integration, not the learned auxiliary models.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from world_models.backends.cosmos import apply_guardrail_compatibility_patch, to_conditioning
from world_models.flow import sample_cosmos_latents
from world_models.observation import Observation


CHECKPOINT = "nvidia/Cosmos-Predict2.5-2B"
REVISION = "0d37c7498f54cee3c599d438d895a0a4a8608064"


def load_pipeline(*, implementation: str = "scratch"):
    """Load the pinned weights on CPU; install our Transformer when selected."""
    from diffusers import Cosmos2_5_PredictBasePipeline
    from world_models.models.cosmos_transformer import ScratchCosmosTransformer

    if implementation not in {"reference", "scratch"}:
        raise ValueError("implementation must be reference or scratch")
    apply_guardrail_compatibility_patch()
    pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
        CHECKPOINT, revision=REVISION, torch_dtype=torch.bfloat16,
    )
    if implementation == "scratch":
        pipeline.register_modules(transformer=ScratchCosmosTransformer.from_reference(pipeline.transformer))
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


@torch.inference_mode()
def encode_text(pipeline, prompt: str, *, cache: Path | None = None,
                device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Encode once before video inference so the text model can return to CPU."""
    import hashlib

    identity = hashlib.sha256((CHECKPOINT + REVISION + prompt + "max_length=512").encode()).hexdigest()
    if cache is not None and cache.exists():
        record = torch.load(cache, map_location="cpu", weights_only=True)
        if record["identity"] != identity:
            raise ValueError("Text embedding cache belongs to a different prompt/checkpoint")
        return record["positive"].to(device), record["negative"].to(device)
    pipeline.text_encoder.to(device)
    positive, negative = pipeline.encode_prompt(prompt=prompt, device=torch.device(device),
                                                do_classifier_free_guidance=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": identity, "positive": positive.cpu(), "negative": negative.cpu()}, cache)
    pipeline.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return positive, negative


@torch.inference_mode()
def generate_video(
    pipeline,
    observation: Observation,
    *,
    prompt: str,
    implementation: str,
    seed: int = 0,
    num_frames: int = 29,
    num_steps: int = 15,
    guidance_scale: float = 7.0,
    height: int = 704,
    width: int = 1280,
    offload: bool = True,
    text_cache: Path | None = None,
    callback: Callable[[int, float, torch.Tensor], None] | None = None,
) -> np.ndarray:
    """Produce an RGB uint8 video using identical input conventions for both implementations."""
    if num_frames < 5 or (num_frames - 1) % 4:
        raise ValueError("Use at least five output frames on the 4k+1 VAE grid")
    if observation.num_frames < 5:
        raise ValueError("The chapter's Video2World exercise uses five observed frames")
    if height % 16 or width % 16:
        raise ValueError("height and width must be divisible by 16")
    device = torch.device("cuda")
    # Preserve the safety checks used for the Chapter 2 pipeline.
    pipeline.safety_checker.to(device)
    if not pipeline.safety_checker.check_text_safety(prompt):
        raise ValueError("The Cosmos checker declined this prompt")
    pipeline.safety_checker.to("cpu")
    torch.cuda.empty_cache()
    positive, negative = encode_text(pipeline, prompt, cache=text_cache)
    if offload:
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device)
    conditioning = to_conditioning(observation.last(5))
    generator = torch.Generator(device=device).manual_seed(seed)
    if implementation == "reference":
        def capture(_pipeline, step, timestep, values):
            if callback is not None:
                callback(step, float(timestep), values["latents"])
            return values
        output = pipeline(
            video=conditioning, prompt=None, prompt_embeds=positive,
            negative_prompt_embeds=negative,
            num_frames=num_frames, height=height, width=width,
            num_inference_steps=num_steps, guidance_scale=guidance_scale,
            generator=generator, output_type="np", callback_on_step_end=capture,
        )
        return np.rint(np.clip(output.frames[0], 0, 1) * 255).astype(np.uint8)

    if implementation != "scratch":
        raise ValueError("implementation must be reference or scratch")
    dtype = pipeline.vae.dtype
    video = pipeline.video_processor.preprocess_video(conditioning, height, width)
    pad = video[:, :, -1:].repeat(1, 1, num_frames - 5, 1, 1)
    video = torch.cat([video, pad], dim=2).to(device=device, dtype=dtype)
    latents, known, mask, _ = pipeline.prepare_latents(
        video=video, batch_size=1, num_channels_latents=16, height=height, width=width,
        num_frames_in=5, num_frames_out=num_frames, dtype=torch.float32,
        device=device, generator=generator,
    )
    # Video encoding precedes denoising; free its weights while the DiT runs.
    if offload:
        pipeline.vae.to("cpu")
        torch.cuda.empty_cache()
    padding = latents.new_zeros(1, 1, height, width, dtype=pipeline.transformer.dtype)
    latents = sample_cosmos_latents(
        pipeline.transformer, pipeline.scheduler, latents, known, mask, positive, negative,
        num_steps=num_steps, guidance_scale=guidance_scale, padding_mask=padding, callback=callback,
    )
    mean = pipeline.latents_mean.to(latents.device, latents.dtype)
    inverse_std = pipeline.latents_std.to(latents.device, latents.dtype)
    decoded = pipeline.vae.decode((latents / inverse_std + mean).to(dtype), return_dict=False)[0]
    if decoded.shape[2] != num_frames:
        raise ValueError(f"VAE decoded {decoded.shape[2]} frames; expected {num_frames}")
    frames = pipeline.video_processor.postprocess_video(decoded, output_type="np")[0]
    # Match the official pre-check conversion, including truncation to uint8.
    frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    pipeline.safety_checker.to(device)
    frames = pipeline.safety_checker.check_video_safety(frames)
    pipeline.maybe_free_model_hooks()
    return np.asarray(frames, dtype=np.uint8)
