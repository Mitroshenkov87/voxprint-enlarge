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

from enlarge.appinfo import APP_DISPLAY_NAME
from enlarge.ipc.capabilities import ART, PHOTO_FAST, PHOTO_MAX
from enlarge.ipc.naming import peek_default, peek_explicit, reserve_default, reserve_explicit
from enlarge.ipc.protocol import (
    STAGES,
    EnlargeRequest,
    ErrorCode,
    Stage,
    accepted_message,
    done_message,
    error_message,
    exit_code,
    progress_message,
    warning_message,
)
from enlarge.ipc.scale import choose_scale, planned_size
from enlarge.suite import gpu
from voxprint_common.gpu_lock import GpuLockTimeout, gpu_job, read_holder
from voxprint_engine.nvidia_smi import assess
from voxprint_engine.formats import (
    DEFAULT_FORMAT,
    open_rgb,
    output_extension,
    require_output_format,
    save_image,
)
from voxprint_engine.providers import ci_lanczos
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


# A first TensorRT build on the 4090 took about 90 seconds. A linear ETA from
# the first progress fraction reported a few seconds and then sat there.
_FIRST_BUILD_S = 90.0


def _stage_for(fraction: float, message: str = "") -> Stage:
    """Map a worker fraction and its text onto a protocol stage.

    ``building_engine`` and ``loading_cached_engine`` are additive. Older
    readers ignore stage strings they do not know. Conversion text stays in
    ``loading`` even when its fraction is high, so a cached ONNX file does
    not report ``encoding`` before the tiles start.
    """
    text = message.lower()
    if "building the tensorrt engine" in text:
        return "building_engine"
    if "loading the cached tensorrt engine" in text or "tensorrt engine is cached" in text:
        return "loading_cached_engine"
    if (
        "onnx file is cached" in text
        or "checked the onnx" in text
        or "converting" in text
        or "loading the weight" in text
        or "checking the model" in text
    ):
        return "loading"
    if fraction < 0.15:
        return "loading"
    if fraction < 0.9:
        return "tiles"
    return "encoding"


def _stage_rank(stage: Stage) -> int:
    """Protocol order. A later line must not name an earlier stage."""
    return STAGES.index(stage)


def _advance(previous: tuple[Stage, float] | None, stage: Stage, pct: float) -> tuple[Stage, float] | None:
    """The next progress point, or None when the stage or the percent would move backwards."""
    if previous is None:
        return stage, pct
    old_stage, old_pct = previous
    if _stage_rank(stage) < _stage_rank(old_stage):
        return None
    if _stage_rank(stage) == _stage_rank(old_stage) and pct + 1e-6 < old_pct:
        return None
    return stage, pct


def _eta_s(pct: float, stage: Stage, started: float, now: float | None = None) -> float | None:
    """Seconds remaining. The first engine build uses a 90 second floor."""
    if pct >= 100.0:
        return 0.0
    if pct < 1.0:
        return None
    elapsed = (time.perf_counter() if now is None else now) - started
    if stage == "building_engine":
        return max(0.0, _FIRST_BUILD_S - elapsed)
    return elapsed * (100.0 - pct) / pct


def _classify(message: str) -> ErrorCode:
    text = message.lower()
    if "cancel" in text:
        return "cancelled"
    if "nvidia gpu" in text or "600 series" in text:
        return "gpu_unsupported"
    if "onnx conversion failed" in text or "onnx export failed" in text or text.startswith("worker closed the pipe"):
        return "model_missing"
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


def parent_lock_matches(pid: int) -> bool:
    """True when the suite lock file exists and its pid is ``pid``."""
    if pid <= 0:
        return False
    holder = read_holder()
    return holder is not None and holder.pid == pid


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
    emit(progress_message(job_id, pct, stage, eta_s=_eta_s(pct, stage, started), lock_holder=lock_holder))


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
        reason = "" if ci_lanczos() else assess(APP_DISPLAY_NAME).message
        if reason and not ci_lanczos():
            return _fail(emit, job_id, "gpu_unsupported", reason, None)
        emit(done_message(job_id, str(destination), planned[0], planned[1], image_format, 0.0))
        return 0

    started = time.perf_counter()
    _emit_progress(emit, job_id, 0, "queued", started)
    if control.is_cancelled():
        return _fail(emit, job_id, "cancelled", "cancelled", destination)

    reason = "" if ci_lanczos() else assess(APP_DISPLAY_NAME).message
    if reason and not ci_lanczos():
        return _fail(emit, job_id, "gpu_unsupported", reason, destination)

    try:
        with _parent_env(request.parent_lock_pid):
            with gpu_job(
                "enlarge",
                f"ipc:{job_id}",
                eta=max(60.0, request.lock_timeout or 60.0),
                timeout=request.lock_timeout,
                poll=0.05,
                cancel=control.is_cancelled,
                on_wait=lambda holder, _waited: _emit_progress(
                    emit, job_id, 1, "waiting_gpu", started, lock_holder=holder.as_dict()
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
                    app_name=APP_DISPLAY_NAME,
                )
                client = WorkerClient()
                control.client = client
                try:
                    with client:
                        if control.is_cancelled():
                            return _fail(emit, job_id, "cancelled", "cancelled", destination)

                        seen: tuple[Stage, float] | None = None

                        def _progress(fraction: float, message: str) -> None:
                            nonlocal seen
                            stage = _stage_for(fraction, message)
                            nxt = _advance(seen, stage, fraction * 100.0)
                            if nxt is None:
                                return
                            seen = nxt
                            _emit_progress(emit, job_id, nxt[1], nxt[0], started)

                        def _warning(code: str, text: str) -> None:
                            emit(warning_message(job_id, code, text))

                        result = client.upscale(upscale, on_progress=_progress, on_warning=_warning)
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
