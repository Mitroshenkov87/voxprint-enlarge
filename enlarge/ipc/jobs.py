"""Run one hand-off job and emit protocol messages.

The server process holds the suite GPU lock for the job. The worker subprocess
sees that lock and does its inference without taking a second one. A caller
that already holds the lock passes ``parent_lock_pid`` and Enlarge runs inside
that file without waiting or releasing it.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

from enlarge.ipc.capabilities import ART, PHOTO_FAST, PHOTO_MAX, gpu_capability
from enlarge.ipc.naming import peek_default, peek_explicit, reserve_default, reserve_explicit
from enlarge.ipc.protocol import (
    EnlargeRequest,
    ErrorCode,
    Stage,
    accepted_message,
    done_message,
    error_message,
    exit_code,
    progress_message,
)
from enlarge.ipc.scale import choose_scale, planned_size
from enlarge.suite import gpu
from voxprint_engine.formats import (
    DEFAULT_FORMAT,
    open_rgb,
    output_extension,
    require_output_format,
    save_image,
)
from voxprint_engine.gpu_lock import GpuLockTimeout, gpu_job, parent_lock_matches
from voxprint_engine.hardware import ci_lanczos
from voxprint_engine.registry import load_builtin
from voxprint_engine.store import weight_report
from voxprint_engine.worker import UpscaleRequest, WorkerClient, WorkerError
from voxprint_engine.worker.lanczos import IMAGE_EXTENSIONS

Emit = Callable[[dict[str, Any]], None]
_PARENT_ENV = "VOXPRINT_GPU_PARENT_PID"


class JobControl:
    """Shared cancel flag and the worker process a cancel request can stop."""

    def __init__(self) -> None:
        """Start idle, with no worker to interrupt."""
        self.cancelled = False
        self.client: WorkerClient | None = None

    def cancel(self) -> None:
        """Mark the job cancelled and interrupt its worker."""
        self.cancelled = True
        client = self.client
        if client is not None:
            client.interrupt()

    def is_cancelled(self) -> bool:
        """True after cancel, including a signal handler."""
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
    if "unsupported" in text or "png is an input" in text or "lossless output" in text:
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
    return exit_code(code)


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return int(image.size[0]), int(image.size[1])
    except (OSError, ValueError):
        frame = open_rgb(path)
        return frame.size


@contextmanager
def _parent_env(pid: int) -> Iterator[None]:
    """Publish the caller's lock pid for this process and the worker it spawns."""
    if pid <= 0:
        yield
        return
    previous = os.environ.get(_PARENT_ENV)
    os.environ[_PARENT_ENV] = str(pid)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_PARENT_ENV, None)
        else:
            os.environ[_PARENT_ENV] = previous


def _destination(request: EnlargeRequest, extension: str, scale: int, *, create: bool) -> Path:
    source = Path(request.input)
    if request.output:
        path = Path(request.output)
        return reserve_explicit(path) if create else peek_explicit(path)
    folder = Path(request.output_dir) if request.output_dir else None
    if create:
        return reserve_default(source, scale, extension, folder)
    return peek_default(source, scale, extension, folder)


def _emit_progress(
    emit: Emit,
    job_id: str,
    pct: float,
    stage: Stage,
    started: float,
    lock_holder: dict[str, Any] | None = None,
) -> None:
    eta_s = None
    if pct >= 1.0:
        elapsed = time.perf_counter() - started
        eta_s = elapsed * (100.0 - pct) / pct
    emit(progress_message(job_id, pct, stage, eta_s=eta_s, lock_holder=lock_holder))


def run_job(
    request: EnlargeRequest,
    emit: Emit,
    control: JobControl,
    *,
    queue_pos: int = 0,
    dry_run: bool = False,
) -> int:
    """Emit ``accepted``, ``progress``, and either ``done`` or ``error``. Return a process exit code.

    ``dry_run`` validates the same inputs and prints ``accepted`` then ``done``
    with the planned path and size. It does not reserve a file, load a model, or use the GPU.
    """
    job_id = request.id
    quality = request.quality.lower()
    preset = request.preset.lower()
    if quality not in {"fast", "max"}:
        return _fail(emit, job_id, "bad_input", "quality must be fast or max", None)
    if request.encode_quality < 1 or request.encode_quality > 100:
        return _fail(emit, job_id, "bad_input", "quality must be from 1 to 100", None)
    if request.lock_timeout < 0:
        return _fail(emit, job_id, "bad_input", "lock timeout must be zero or more", None)
    source = Path(request.input)
    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        return _fail(emit, job_id, "unsupported_format", f"unsupported image type {source.suffix or '(none)'}", None)
    if not source.is_file():
        return _fail(emit, job_id, "bad_input", "source image was not found", None)
    try:
        image_format = require_output_format(request.format or DEFAULT_FORMAT, lossless=request.lossless)
        extension = output_extension(image_format)
    except ValueError as exc:
        return _fail(emit, job_id, "unsupported_format", str(exc), None)
    try:
        width, height = _image_size(source)
        if request.long_side:
            scale = choose_scale(width, height, request.long_side)
        elif request.scale in (2, 4):
            scale = request.scale
        else:
            return _fail(emit, job_id, "bad_input", "scale must be 2 or 4", None)
        planned = planned_size(width, height, scale, request.long_side)
    except (OSError, ValueError) as exc:
        return _fail(emit, job_id, "bad_input", str(exc), None)
    model_id = resolve_model_id(preset, quality, request.model)
    if not model_id:
        return _fail(emit, job_id, "bad_input", f"unknown preset {preset or '(none)'}", None)
    catalog = load_builtin(release=True)
    model = catalog.get(model_id)
    if model is None or not model.selectable:
        return _fail(emit, job_id, "model_missing", f"unknown or unavailable model {model_id}", None)
    if not ci_lanczos() and weight_report(model)[0] != "ready":
        return _fail(emit, job_id, "model_missing", f"{model.id} is not downloaded", None)
    if request.parent_lock_pid and not parent_lock_matches(request.parent_lock_pid):
        return _fail(emit, job_id, "bad_input", "the suite GPU lock pid does not match --parent-lock-pid", None)
    try:
        destination = _destination(request, extension, scale, create=not dry_run)
    except OSError as exc:
        return _fail(emit, job_id, "io_error", str(exc), None)

    emit(accepted_message(job_id, queue_pos))
    if dry_run:
        if control.is_cancelled():
            return _fail(emit, job_id, "cancelled", "cancelled", None)
        report = gpu_capability()
        if report["status"] == "none" and not ci_lanczos():
            return _fail(
                emit,
                job_id,
                "gpu_unsupported",
                "Enlarge needs an NVIDIA GPU that CUDA can use, on a driver from the 600 series or newer.",
                None,
            )
        emit(done_message(job_id, str(destination), planned[0], planned[1], image_format, 0.0))
        return 0

    started = time.perf_counter()
    _emit_progress(emit, job_id, 0, "queued", started)
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

    try:
        with _parent_env(request.parent_lock_pid):
            with gpu_job(
                f"enlarge:ipc:{job_id}",
                eta_s=max(60.0, request.lock_timeout or 60.0),
                timeout=request.lock_timeout,
                poll=0.05,
                cancel=control.is_cancelled,
                on_wait=lambda holder, _waited: _emit_progress(
                    emit, job_id, 1, "waiting_gpu", started, lock_holder=holder
                ),
            ):
                if control.is_cancelled():
                    return _fail(emit, job_id, "cancelled", "cancelled", destination)
                _emit_progress(emit, job_id, 5, "loading", started)
                upscale = UpscaleRequest(
                    id=job_id,
                    src=str(source),
                    dst=str(destination),
                    scale=scale,
                    format=image_format,
                    quality=request.encode_quality,
                    model_id=model.id,
                    tile=0,
                    model_tile=model.tile,
                    device=gpu(),
                    lossless=request.lossless,
                )
                client = WorkerClient()
                control.client = client
                try:
                    with client:
                        if control.is_cancelled():
                            return _fail(emit, job_id, "cancelled", "cancelled", destination)

                        def _progress(fraction: float, _message: str) -> None:
                            _emit_progress(emit, job_id, fraction * 100.0, _stage_for(fraction), started)

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
    out_w, out_h = result.width, result.height
    if request.long_side and (out_w, out_h) != planned:
        try:
            frame = open_rgb(destination)
            resized = frame.resize(planned, Image.Resampling.LANCZOS)
            save_image(resized, destination, image_format, request.encode_quality, request.lossless)
            out_w, out_h = planned
        except (OSError, ValueError) as exc:
            return _fail(emit, job_id, "io_error", str(exc), destination)
        if control.is_cancelled():
            return _fail(emit, job_id, "cancelled", "cancelled", destination)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _emit_progress(emit, job_id, 100, "encoding", started)
    emit(done_message(job_id, str(destination), out_w, out_h, image_format, elapsed_ms))
    return 0
