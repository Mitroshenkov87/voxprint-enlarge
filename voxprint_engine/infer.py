"""Run one image through ONNX, then PyTorch.

The worker calls :func:`upscale_file`. It downloads weights before taking the
GPU lock, then passes ``weight`` so the locked section does not touch the network.
``onnx_path`` overrides the sibling ``.onnx`` and is how the smoke test runs
without a real weight file.

``VOXPRINT_CI_LANCZOS=1`` resamples with Lanczos and skips the download. That
flag is for tests and CI. A user job without an NVIDIA provider raises
:class:`~voxprint_engine.errors.ImageJobError`.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from voxprint_engine.formats import open_rgb, require_output_format, save_image
from voxprint_engine.hardware import ci_lanczos
from voxprint_engine.onnx_export import OnnxExportError, cached_onnx, ensure_onnx
from voxprint_engine.providers import TRT, choices_for, cpu_ep_requested, cuda_fallback_choices, gpu_provider_names
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.session import (
    EngineBuildError,
    InferenceError,
    available_providers,
    open_session,
    run_tile,
    trt_build_message,
)
from voxprint_engine.store import StoreError
from voxprint_engine.tile import upscale_tiled
from voxprint_engine.torch_runner import TorchFallbackError, TorchUpscaler, torch_available
from voxprint_engine.vram import choose_tile, halve_tile, is_memory_error, measure, note_spill, overlap_for
from voxprint_engine.worker.lanczos import (
    IMAGE_EXTENSIONS,
    MAX_OUTPUT_PIXELS,
    VIDEO_EXTENSIONS,
    ImageJobError,
    lanczos_resize,
    output_scale,
)

try:
    import tensorrt as _tensorrt
except ImportError:
    _tensorrt = None

Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class JobOutcome:
    """What one image run produced: paths, tile choice, and whether it spilled."""
    width: int
    height: int
    engine: str
    message: str
    tile: int
    overlap: int
    vram_budget_gb: float
    spilled: bool
    next_batch: int
    scale: int


def uses_lanczos(model: ModelManifest, device: str, engine: str = "auto") -> bool:
    """True when this job must not download or open a network."""
    return _lanczos_reason(model, device, engine) != ""


def preflight(
    src: Path,
    dst: Path,
    scale: int,
    image_format: str,
    quality: int,
    double: bool,
    lossless: bool = False,
) -> int:
    """Reject a job before a download or the GPU lock. Returns the output factor."""
    if quality < 1 or quality > 100:
        raise ImageJobError("quality must be from 1 to 100")
    require_output_format(image_format, lossless=lossless)
    if src.resolve() == dst.resolve():
        raise ImageJobError("output path must be different from the source")
    factor = output_scale(scale, double)
    image = _load(src)
    if image.shape[1] * factor * image.shape[0] * factor > MAX_OUTPUT_PIXELS:
        raise ImageJobError("output would be larger than 80 million pixels")
    return factor


def _onnx_cpu(device: str, engine: str) -> bool:
    """The CI smoke test. Not a user mode."""
    return engine == "onnx" or device.strip().lower() == "cpu-onnx" or cpu_ep_requested()


def _lanczos_reason(model: ModelManifest, device: str, engine: str) -> str:
    """Empty when this job should run the network.

    Lanczos runs only while ``VOXPRINT_CI_LANCZOS`` is set. The model id and a
    CPU device do not select it.
    """
    del model
    if _onnx_cpu(device, engine):
        return ""
    if ci_lanczos():
        return "ci"
    return ""


def _passes(factor: int, native: int) -> int:
    """How many native passes are needed so the result is at least ``factor``."""
    if native <= 1:
        return 1
    passes = 1
    produced = native
    while produced < factor and passes < 4:
        passes += 1
        produced *= native
    return passes


def _save(image: np.ndarray, dst: Path, image_format: str, quality: int, lossless: bool) -> None:
    frame = Image.fromarray(np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8), mode="RGB")
    save_image(frame, dst, image_format, quality, lossless)


def _load(src: Path) -> np.ndarray:
    suffix = src.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        raise ImageJobError("Enlarge takes images only. Video belongs to Voxprint AI Clarity.")
    if suffix not in IMAGE_EXTENSIONS:
        raise ImageJobError(f"unsupported image type {suffix or '(none)'}")
    if not src.is_file():
        raise ImageJobError("source image was not found")
    try:
        frame = open_rgb(src)
        return np.asarray(frame, dtype=np.float32) / 255.0
    except (OSError, ValueError) as exc:
        raise ImageJobError(f"could not read the image: {exc}") from exc


def _gpu_label() -> tuple[str, str, str]:
    """``(gpu, driver, trt)`` for the cache folder. Missing tools use plain names."""
    trt = "no-trt"
    if _tensorrt is not None:
        trt = str(getattr(_tensorrt, "__version__", "trt"))
    identity = _nvidia_identity()
    if identity is None:
        return "cpu", "none", trt
    return identity[0], identity[1], trt


def _nvidia_identity() -> tuple[str, str] | None:
    command = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    try:
        if os.name == "nt":
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        else:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _downscale(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Lanczos down to the requested size when a native pass overshot (4x net, 2x ask)."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    frame = Image.fromarray(np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8), mode="RGB")
    frame = frame.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(frame, dtype=np.float32) / 255.0


def _close_session(session: object) -> None:
    closer = getattr(session, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except (OSError, RuntimeError):
        return


def _run_network(
    image: np.ndarray,
    infer: Callable[[np.ndarray], np.ndarray],
    tile: int,
    native: int,
    factor: int,
) -> tuple[np.ndarray, int, bool]:
    """Apply the network until ``factor`` is reached. Halve the tile on an allocator error."""
    spilled = False
    current = tile
    while True:
        try:
            produced = image
            passes = _passes(factor, native)
            grown = 1
            for _ in range(passes):
                produced = upscale_tiled(produced, infer, current, overlap_for(current), native)
                grown *= native
            if grown != factor:
                source_h, source_w = image.shape[0], image.shape[1]
                produced = _downscale(produced, source_w * factor, source_h * factor)
            return produced, current, spilled
        except (InferenceError, RuntimeError, OSError, ValueError) as exc:
            if not is_memory_error(exc):
                raise
            nxt = halve_tile(current)
            if nxt is None:
                raise
            current = nxt
            spilled = True


def _finish(
    image: np.ndarray,
    dst: Path,
    image_format: str,
    quality: int,
    lossless: bool,
    *,
    engine: str,
    message: str,
    tile: int,
    budget_gb: float,
    spilled: bool,
    batch: int,
    factor: int,
    on_progress: Progress | None,
) -> JobOutcome:
    _save(image, dst, image_format, quality, lossless)
    if on_progress is not None:
        on_progress(1.0, message)
    return JobOutcome(
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        engine=engine,
        message=message,
        tile=tile,
        overlap=overlap_for(tile),
        vram_budget_gb=round(budget_gb, 3),
        spilled=spilled,
        next_batch=note_spill(batch, spilled),
        scale=factor,
    )


def upscale_file(
    src: Path,
    dst: Path,
    model: ModelManifest,
    *,
    scale: int,
    image_format: str,
    quality: int,
    double: bool,
    device: str,
    user_tile: int,
    batch: int,
    engine: str = "auto",
    lossless: bool = False,
    on_progress: Progress | None = None,
    models_root: Path | None = None,
    onnx_path: Path | None = None,
    weight: Path | None = None,
) -> JobOutcome:
    """Write ``dst``.

    ``weight``, when set, is the pinned ``.pth`` already checked by the caller.
    ``onnx_path`` overrides the ``.onnx`` sitting next to that file.
    """
    factor = preflight(src, dst, scale, image_format, quality, double, lossless)
    snapshot = measure(device if device.strip().lower() != "cpu-onnx" else "cpu")
    reason = _lanczos_reason(model, device, engine)
    tile = choose_tile(snapshot.budget_gb, model.tile, user_tile)
    if not reason and not _onnx_cpu(device, engine) and not gpu_provider_names(available_providers(), device=device):
        raise ImageJobError(
            "Enlarge needs an NVIDIA GPU that CUDA can use. This computer does not have one."
        )
    if reason:
        if on_progress is not None:
            on_progress(0.4, reason)
        width, height = lanczos_resize(src, dst, scale, image_format, quality, double, lossless)
        return JobOutcome(
            width=width,
            height=height,
            engine="lanczos",
            message=reason,
            tile=tile,
            overlap=overlap_for(tile),
            vram_budget_gb=round(snapshot.budget_gb, 3),
            spilled=False,
            next_batch=note_spill(batch, False),
            scale=factor,
        )

    network = onnx_path or cached_onnx(model, models_root)
    if network is None and weight is not None:
        sibling = weight.with_suffix(".onnx")
        if sibling.is_file():
            network = sibling
    if network is None:
        if on_progress is not None:
            on_progress(0.1, "checking the model file")
        try:
            network = ensure_onnx(model, models_root, on_progress, weight=weight)
        except (StoreError, OnnxExportError) as exc:
            raise ImageJobError(str(exc)) from exc
    image = _load(src)
    gpu, driver, trt = _gpu_label()
    installed = available_providers()
    attempts = [
        choices_for(
            installed,
            device,
            gpu=gpu,
            driver=driver,
            trt=trt,
            execution=model.execution,
            in_channels=model.in_channels,
        )
    ]
    if model.execution.strip().lower() != "cuda":
        fallback = cuda_fallback_choices(
            installed,
            device,
            gpu=gpu,
            driver=driver,
            trt=trt,
            in_channels=model.in_channels,
        )
        if fallback:
            attempts.append(fallback)
    native = model.scale if model.scale > 1 else factor
    ran: np.ndarray | None = None
    used = ""
    spilled = False
    runtime_error: InferenceError | None = None
    if network is not None:
        for choices in attempts:
            if not choices:
                continue
            try:
                if on_progress is not None and choices[0].name == TRT:
                    cache = Path(choices[0].options.get("trt_engine_cache_path") or "")
                    on_progress(0.4, trt_build_message(cache))
                session, provider = open_session(network, choices)
                if on_progress is not None:
                    on_progress(0.5, f"tile {tile}, overlap {overlap_for(tile)}")
            except EngineBuildError:
                continue
            try:

                def _tile(tile_image: np.ndarray) -> np.ndarray:
                    return run_tile(session, tile_image)

                ran, tile, spilled = _run_network(image, _tile, tile, native, factor)
                used = f"onnx {provider}"
            except (InferenceError, RuntimeError, OSError, ValueError) as exc:
                if is_memory_error(exc):
                    raise ImageJobError(f"out of memory at tile {tile}") from exc
                # A TensorRT engine can build and then fail the kernel. Try CUDA.
                if isinstance(exc, InferenceError):
                    runtime_error = exc
                    continue
                raise ImageJobError(str(exc)) from exc
            finally:
                _close_session(session)
            break
    if ran is None:
        if weight is None or not torch_available():
            if runtime_error is not None:
                detail = str(runtime_error)
            elif weight is None:
                detail = "no ONNX file and no PyTorch fallback"
            else:
                detail = "the ONNX session did not build and PyTorch is not installed"
            raise ImageJobError(detail)
        try:
            runner = TorchUpscaler(weight, device)
        except TorchFallbackError as exc:
            raise ImageJobError(str(exc)) from exc
        native = runner.scale or model.scale or factor
        try:
            ran, tile, spilled = _run_network(image, runner, tile, native, factor)
        except (InferenceError, RuntimeError, OSError, ValueError) as exc:
            if is_memory_error(exc):
                raise ImageJobError(f"out of memory at tile {tile}") from exc
            raise ImageJobError(str(exc)) from exc
        used = "torch"
    if ran is None:
        raise ImageJobError("the image was not produced")
    return _finish(
        ran,
        dst,
        image_format,
        quality,
        lossless,
        engine=used.split()[0] if used else "onnx",
        message=used or "onnx",
        tile=tile,
        budget_gb=snapshot.budget_gb,
        spilled=spilled,
        batch=batch,
        factor=factor,
        on_progress=on_progress,
    )
