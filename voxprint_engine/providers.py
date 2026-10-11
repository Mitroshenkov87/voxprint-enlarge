"""Which ONNX Runtime execution provider to try, and where a TensorRT engine goes.

A GPU job opens TensorRT. CUDA is a second attempt only when that engine does
not build, or the session itself when the TensorRT provider is not installed.
DirectML and Windows ML are not used. The CPU execution provider is the smoke
test (``VOXPRINT_ORT_DEVICE=cpu`` or ``device`` ``cpu-onnx``), not a user mode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, assert_never

from voxprint_common.paths import trt_cache_dir
from voxprint_engine.registry.schema import checked_in_channels
from voxprint_engine.vram import device_index

TRT = "TensorrtExecutionProvider"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"

Execution = Literal["auto", "cuda", "tensorrt"]


@dataclass(frozen=True)
class ProviderChoice:
    """One ONNX Runtime provider and the options passed to it."""
    name: str
    options: dict[str, str]


def cpu_ep_requested() -> bool:
    """The smoke test and a local CPU benchmark set ``VOXPRINT_ORT_DEVICE=cpu``."""
    return os.environ.get("VOXPRINT_ORT_DEVICE", "").strip().lower() == "cpu"


def ci_lanczos() -> bool:
    """True only for the internal test flag. Users cannot turn this on from the app."""
    return os.environ.get("VOXPRINT_CI_LANCZOS", "").strip() == "1"


def _clean_device(device: str) -> str:
    return (device or "auto").strip().lower()


def gpu_provider_names(available: set[str] | list[str], *, device: str = "auto") -> list[str]:
    """TensorRT, then CUDA. DirectML, Windows ML, and CPU are left out.

    CUDA is present in this list so a failed TensorRT build has a second
    attempt. :func:`choices_for` opens only the first name.
    """
    if _clean_device(device) == "cpu":
        return []
    have = set(available)
    names: list[str] = []
    if TRT in have:
        names.append(TRT)
    if CUDA in have:
        names.append(CUDA)
    return names


def provider_options(
    name: str,
    device: str,
    *,
    gpu: str,
    driver: str,
    trt: str,
    in_channels: int = 3,
) -> dict[str, str]:
    """fp16 for TensorRT, plus a rebuildable engine cache. CUDA gets a device id.

    The TensorRT profile fixes the channel count from the manifest. Height and
    width stay dynamic. The input name is ``input``, which is what the
    exporter writes.
    """
    index = str(device_index(device))
    if name == TRT:
        channels = checked_in_channels(in_channels)
        del gpu, driver, trt
        folder = trt_cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        return {
            "device_id": index,
            "trt_fp16_enable": "True",
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": str(folder),
            "trt_profile_min_shapes": f"input:1x{channels}x8x8",
            "trt_profile_opt_shapes": f"input:1x{channels}x256x256",
            "trt_profile_max_shapes": f"input:1x{channels}x2048x2048",
        }
    if name == CUDA:
        return {"device_id": index}
    return {}


def _choice(name: str, device: str, *, gpu: str, driver: str, trt: str, in_channels: int = 3) -> ProviderChoice:
    return ProviderChoice(
        name,
        provider_options(name, device, gpu=gpu, driver=driver, trt=trt, in_channels=in_channels),
    )


def execution_mode(value: str) -> Execution:
    """``auto``, ``cuda``, or ``tensorrt``. Anything else is rejected."""
    mode = (value or "auto").strip().lower()
    if mode == "auto":
        return "auto"
    if mode == "cuda":
        return "cuda"
    if mode == "tensorrt":
        return "tensorrt"
    raise ValueError(f"execution must be auto, cuda, or tensorrt, not {value!r}")


def choices_for(
    available: set[str] | list[str],
    device: str,
    *,
    gpu: str = "gpu",
    driver: str = "driver",
    trt: str = "no-trt",
    execution: str = "auto",
    in_channels: int = 3,
) -> list[ProviderChoice]:
    """The session to open first.

    ``auto`` is TensorRT, or CUDA when TensorRT is absent. ``cuda`` and
    ``tensorrt`` stay on that provider. A model TensorRT cannot execute
    (the DAT2 photo model) is marked ``cuda`` in the catalogue.
    ``in_channels`` is the manifest field. TensorRT profiles use it.
    """
    checked_in_channels(in_channels)
    if cpu_ep_requested() or _clean_device(device) == "cpu-onnx":
        return [ProviderChoice(CPU, {})]
    names = gpu_provider_names(available, device=device)
    mode = execution_mode(execution)
    if mode == "auto":
        if not names:
            return []
        return [_choice(names[0], device, gpu=gpu, driver=driver, trt=trt, in_channels=in_channels)]
    if mode == "cuda":
        if CUDA not in names:
            return []
        return [_choice(CUDA, device, gpu=gpu, driver=driver, trt=trt, in_channels=in_channels)]
    if mode == "tensorrt":
        if TRT not in names:
            return []
        return [_choice(TRT, device, gpu=gpu, driver=driver, trt=trt, in_channels=in_channels)]
    assert_never(mode)


def cuda_fallback_choices(
    available: set[str] | list[str],
    device: str,
    *,
    gpu: str = "gpu",
    driver: str = "driver",
    trt: str = "no-trt",
    in_channels: int = 3,
) -> list[ProviderChoice]:
    """CUDA after a TensorRT engine failed to build. Empty when CUDA was already the session."""
    checked_in_channels(in_channels)
    if cpu_ep_requested() or _clean_device(device) in {"cpu", "cpu-onnx"}:
        return []
    names = gpu_provider_names(available, device=device)
    if names[:1] != [TRT] or CUDA not in names:
        return []
    return [_choice(CUDA, device, gpu=gpu, driver=driver, trt=trt, in_channels=in_channels)]
