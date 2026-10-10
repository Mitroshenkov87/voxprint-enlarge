"""Run one hand-off job and emit protocol messages.

The server process holds the suite GPU lock for the job. The worker subprocess
sees that lock and does its inference without taking a second one.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from enlarge.ipc.capabilities import ART, PHOTO_FAST, PHOTO_MAX, gpu_capability
from enlarge.ipc.naming import reserve_default, reserve_explicit
from enlarge.ipc.protocol import (
    EnlargeRequest,
    ErrorCode,
    Stage,
    accepted_message,
    done_message,
    error_message,
    progress_message,
)
from enlarge.suite import gpu
from voxprint_engine.formats import DEFAULT_FORMAT, DEFAULT_QUALITY, output_extension, require_output_format
from voxprint_engine.gpu_lock import GpuLockTimeout, gpu_job
from voxprint_engine.hardware import ci_lanczos
from voxprint_engine.registry import load_builtin
from voxprint_engine.worker import UpscaleRequest, WorkerClient, WorkerError
from voxprint_engine.worker.lanczos import IMAGE_EXTENSIONS

Emit = Callable[[dict[str, Any]], None]
Cancel = Callable[[], bool]


class JobControl:
    """Shared cancel flag and the worker process a cancel request can stop."""

    def __init__(self) -> None:
        self.cancelled = False
        self.client: WorkerClient | None = None

    def cancel(self) -> None:
        self.cancelled = True
        client = self.client
        if client is not None:
            client.interrupt()

    def is_cancelled(self) -> bool:
        return self.cancelled


def resolve_model_id(preset: str, quality: str, model: str) -> str:
    """An explicit model id wins. Otherwise preset and quality pick a pinned model."""
    if model:
        return model
    if preset == "photo" and quality == "max":
        return PHOTO_MAX
    if preset == "photo" and quality == "fast":
        return PHOTO_FAST
    if preset == "art" and quality in {"fast", "max"}:
        return ART
    return ""


def _stage_for(fraction: float) -> Stage:
    if fraction < 0.15:
        return "loading"
    if fraction < 0.9:
        return "tiles"
    return "encoding"


def _classify(message: str) -> ErrorCode:
    text = message.lower()
    if "cancel" in text:
        return "cancelled"
    if "nvidia gpu" in text or "600 series" in text:
        return "gpu_unsupported"
    if "out of memory" in text:
        return "oom"
    if "unsupported" in text or "png is an input" in text:
        return "unsupported_format"
    if "not found" in text or "could not read" in text or "could not resize" in text:
        return "bad_input"
    if "model" in text:
        return "model_missing"
    if "busy" in text:
        return "gpu_busy"
    if "permission" in text or "no space" in text or "read-only" in text:
        return "io_error"
    return "internal"


def _fail(emit: Emit, job_id: str, code: ErrorCode, message: str, placeholder: Path | None) -> int:
    if placeholder is not None:
        try:
            placeholder.unlink()
        except OSError:
            pass
    emit(error_message(job_id, code, message))
    return 1


def _destination(request: EnlargeRequest, extension: str) -> Path:
    source = Path(request.input)
    if request.output:
        return reserve_explicit(Path(request.output))
    folder = Path(request.output_dir) if request.output_dir else None
    return reserve_default(source, request.scale, extension, folder)


def run_job(request: EnlargeRequest, emit: Emit, control: JobControl, *, queue_pos: int = 0) -> int:
    """Emit ``accepted``, ``progress``, and either ``done`` or ``error``. Return a process exit code."""
    job_id = request.id
    quality = request.quality.lower()
    preset = request.preset.lower()
    if request.scale not in (2, 4):
        return _fail(emit, job_id, "bad_input", "scale must be 2 or 4", None)
    if quality not in {"fast", "max"}:
        return _fail(emit, job_id, "bad_input", "quality must be fast or max", None)
    source = Path(request.input)
    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        return _fail(emit, job_id, "unsupported_format", f"unsupported image type {source.suffix or '(none)'}", None)
    if not source.is_file():
        return _fail(emit, job_id, "bad_input", "source image was not found", None)
    try:
        image_format = require_output_format(request.format or DEFAULT_FORMAT, lossless=False)
        extension = output_extension(image_format)
    except ValueError as exc:
        return _fail(emit, job_id, "unsupported_format", str(exc), None)
    model_id = resolve_model_id(preset, quality, request.model)
    if not model_id:
        return _fail(emit, job_id, "bad_input", f"unknown preset {preset or '(none)'}", None)
    catalog = load_builtin(release=True)
    model = catalog.get(model_id)
    if model is None or not model.selectable:
        return _fail(emit, job_id, "model_missing", f"unknown or unavailable model {model_id}", None)
    try:
        destination = _destination(request, extension)
    except OSError as exc:
        return _fail(emit, job_id, "io_error", str(exc), None)

    emit(accepted_message(job_id, queue_pos))
    emit(progress_message(job_id, 0, "queued"))
    if control.is_cancelled():
        return _fail(emit, job_id, "cancelled", "cancelled", destination)

    report = gpu_capability()
    if report["status"] == "none" and not ci_lanczos():
        return _fail(
            emit,
            job_id,
            "gpu_unsupported",
            "Enlarge needs an NVIDIA GPU that CUDA can use, on a driver from the 600 series or newer.",
            destination,
        )

    started = time.perf_counter()
    try:
        with gpu_job(
            f"enlarge:ipc:{job_id}",
            eta_s=600.0,
            timeout=2.0,
            poll=0.05,
            cancel=control.is_cancelled,
            on_wait=lambda *_args: emit(progress_message(job_id, 1, "waiting_gpu")),
        ):
            if control.is_cancelled():
                return _fail(emit, job_id, "cancelled", "cancelled", destination)
            emit(progress_message(job_id, 5, "loading"))
            upscale = UpscaleRequest(
                id=job_id,
                src=str(source),
                dst=str(destination),
                scale=request.scale,
                format=image_format,
                quality=DEFAULT_QUALITY,
                model_id=model.id,
                tile=0,
                model_tile=model.tile,
                device=gpu(),
            )
            client = WorkerClient()
            control.client = client
            try:
                with client:
                    if control.is_cancelled():
                        return _fail(emit, job_id, "cancelled", "cancelled", destination)

                    def _progress(fraction: float, _message: str) -> None:
                        emit(progress_message(job_id, fraction * 100.0, _stage_for(fraction)))

                    result = client.upscale(upscale, on_progress=_progress)
            finally:
                control.client = None
    except GpuLockTimeout as exc:
        code: ErrorCode = "cancelled" if control.is_cancelled() else "gpu_busy"
        text = "cancelled" if code == "cancelled" else "the GPU is busy"
        if code == "gpu_busy" and str(exc):
            text = str(exc)
        return _fail(emit, job_id, code, text, destination)
    except WorkerError as exc:
        if control.is_cancelled():
            return _fail(emit, job_id, "cancelled", "cancelled", destination)
        code = _classify(str(exc))
        return _fail(emit, job_id, code, str(exc), destination)
    except OSError as exc:
        return _fail(emit, job_id, "io_error", str(exc), destination)

    if control.is_cancelled():
        return _fail(emit, job_id, "cancelled", "cancelled", destination)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    emit(progress_message(job_id, 100, "encoding"))
    emit(
        done_message(
            job_id,
            str(destination),
            result.width,
            result.height,
            image_format,
            elapsed_ms,
        )
    )
    return 0
