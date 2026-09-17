from __future__ import annotations

import os
from dataclasses import dataclass

import torch


# ============================================================
# V0.3.4-B3.1.1
# Compute Device Abstraction
# ============================================================
#
# Goal:
#   - Local NVIDIA GPU available -> use CUDA automatically.
#   - Cloud / CPU-only machine     -> fall back to CPU automatically.
#   - Optional manual override via .env:
#
#       COMPUTE_DEVICE=auto   # recommended
#       COMPUTE_DEVICE=cuda
#       COMPUTE_DEVICE=cpu
#
# No module should hard-code "cpu" or "cuda" after this stage.
# ============================================================


VALID_DEVICE_MODES = {
    "auto",
    "cuda",
    "cpu",
}


@dataclass(frozen=True)
class DeviceInfo:
    requested_mode: str
    device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    torch_version: str
    torch_cuda_version: str | None
    fallback_used: bool

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")


def _requested_mode() -> str:
    value = (
        os.getenv(
            "COMPUTE_DEVICE",
            "auto",
        )
        .strip()
        .lower()
    )

    if value not in VALID_DEVICE_MODES:
        raise ValueError(
            "Invalid COMPUTE_DEVICE="
            f"{value!r}. "
            "Allowed values: auto, cuda, cpu."
        )

    return value


def get_device_info() -> DeviceInfo:
    requested = _requested_mode()

    cuda_available = bool(
        torch.cuda.is_available()
    )

    cuda_device_count = (
        torch.cuda.device_count()
        if cuda_available
        else 0
    )

    cuda_device_name = None

    if cuda_available:
        try:
            cuda_device_name = (
                torch.cuda.get_device_name(0)
            )
        except Exception:
            cuda_device_name = "CUDA GPU"

    fallback_used = False

    if requested == "cpu":
        device = "cpu"

    elif requested == "cuda":
        if cuda_available:
            device = "cuda"
        else:
            # User explicitly requested CUDA, but portability
            # is more important than crashing the whole Agent.
            device = "cpu"
            fallback_used = True

    else:
        # Recommended mode.
        device = (
            "cuda"
            if cuda_available
            else "cpu"
        )

    return DeviceInfo(
        requested_mode=requested,
        device=device,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        torch_version=str(
            torch.__version__
        ),
        torch_cuda_version=(
            str(torch.version.cuda)
            if torch.version.cuda
            else None
        ),
        fallback_used=fallback_used,
    )


def get_compute_device() -> str:
    return get_device_info().device


def get_embedding_device() -> str:
    """
    SentenceTransformer / HuggingFaceEmbeddings
    accepts 'cuda' / 'cpu'.
    """
    return get_compute_device()


def get_vision_device() -> str:
    return get_compute_device()


def get_vision_dtype():
    """
    FP16 is appropriate on CUDA for this Agent's local
    inference path. CPU uses FP32 for compatibility.
    """
    info = get_device_info()

    if info.is_cuda:
        return torch.float16

    return torch.float32


def print_device_report(
    prefix: str = "[DEVICE]",
) -> None:
    info = get_device_info()

    print(
        f"{prefix} requested="
        f"{info.requested_mode}"
    )

    print(
        f"{prefix} selected="
        f"{info.device}"
    )

    print(
        f"{prefix} torch="
        f"{info.torch_version}"
    )

    print(
        f"{prefix} torch CUDA runtime="
        f"{info.torch_cuda_version}"
    )

    print(
        f"{prefix} cuda_available="
        f"{info.cuda_available}"
    )

    if info.cuda_available:
        print(
            f"{prefix} GPU="
            f"{info.cuda_device_name}"
        )

        print(
            f"{prefix} GPU count="
            f"{info.cuda_device_count}"
        )

    if info.fallback_used:
        print(
            f"{prefix} WARNING: CUDA was requested "
            "but is unavailable. Falling back to CPU."
        )
