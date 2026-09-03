"""Environment verification for the world-models repository."""
from __future__ import annotations

import importlib.util
import platform
import shutil
import sys

# Import names, which differ from install names for several packages.
CORE_REQUIRED = ("numpy", "imageio", "PIL", "matplotlib")
GPU_REQUIRED = (
    "torch",
    "huggingface_hub",
    "diffusers",
    "transformers",
    "cosmos_guardrail",
    "accelerate",
)


def missing(packages: tuple[str, ...]) -> list[str]:
    """Return the subset of ``packages`` that cannot be imported."""
    return [p for p in packages if importlib.util.find_spec(p) is None]


def require_local_inference() -> None:
    """Raise with an actionable message if the full profile is unavailable."""
    absent = missing(GPU_REQUIRED)
    if absent:
        raise RuntimeError(
            f"Missing required packages: {', '.join(absent)}. "
            "Run: pip install -r requirements-chapter-02-gpu.txt, then "
            "install a CUDA-enabled PyTorch build for the NVIDIA driver."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible to PyTorch. Local inference is unavailable. "
            "Use the artifacts-only profile, or check that the NVIDIA driver "
            "and a CUDA-enabled PyTorch build are installed."
        )


if __name__ == "__main__":
    absent_core = missing(CORE_REQUIRED)
    absent_gpu = missing(GPU_REQUIRED)

    print(f"Python           {platform.python_version()}")
    print(f"Platform         {platform.system()} {platform.machine()}")
    print(f"git-lfs          {'yes' if shutil.which('git-lfs') else 'no'}")

    if importlib.util.find_spec("torch") is not None:
        import torch

        print(f"PyTorch          {torch.__version__}")
        print(f"CUDA available   {torch.cuda.is_available()}")
        print(f"CUDA version     {torch.version.cuda or 'n/a'}")
    else:
        print("PyTorch          not installed")

    if absent_core:
        print(f"MISSING core:    {', '.join(absent_core)}")
    else:
        print("Artifact path    ready")
    if absent_gpu:
        print(f"GPU path missing {', '.join(absent_gpu)}")
    elif not torch.cuda.is_available():
        print("GPU path         packages installed, CUDA unavailable")
    else:
        print("GPU path         ready")

    sys.exit(1 if absent_core else 0)
