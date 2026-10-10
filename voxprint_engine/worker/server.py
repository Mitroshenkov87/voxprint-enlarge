"""Worker main loop. Reads jobs from stdin, writes one JSON line per event to stdout."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from voxprint_engine.formats import DEFAULT_FORMAT
from voxprint_engine.gpu_lock import GpuLockTimeout, gpu_job
from voxprint_engine.infer import JobOutcome, preflight, upscale_file, uses_lanczos
from voxprint_engine.registry import combined_catalog
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.store import StoreError, ensure_weight, register_quietly
from voxprint_engine.worker.lanczos import ImageJobError
from voxprint_engine.worker.restore import handle_restore


_stdout: TextIO = sys.stdout


def emit(message: dict[str, Any], stdout: TextIO | None = None) -> None:
    out = stdout or _stdout
    out.write(json.dumps(message, ensure_ascii=True) + "\n")
    out.flush()


def _as_int(value: object, label: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ImageJobError(f"{label} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ImageJobError(f"{label} must be an integer") from exc


def _model(message: dict[str, Any]) -> ModelManifest:
    model_id = str(message.get("model_id") or "")
    custom = str(message.get("custom_models_dir") or "").strip()
    folder = Path(custom) if custom else None
    catalog = combined_catalog(folder, release=True)
    model = catalog.get(model_id)
    if model is None or not model.selectable:
        raise ImageJobError(f"unknown or unavailable model {model_id or '(none)'}")
    return model


def _result(job_id: str, model_id: str, dst: Path, outcome: JobOutcome) -> dict[str, Any]:
    return {
        "op": "result",
        "id": job_id,
        "ok": True,
        "message": outcome.message,
        "dst": str(dst),
        "width": outcome.width,
        "height": outcome.height,
        "scale": outcome.scale,
        "engine": outcome.engine,
        "model_id": model_id,
        "tile": outcome.tile,
        "overlap": outcome.overlap,
        "vram_budget_gb": outcome.vram_budget_gb,
        "spilled": outcome.spilled,
        "next_batch": outcome.next_batch,
    }


def _job(message: dict[str, Any]) -> dict[str, Any]:
    job_id = str(message.get("id") or "")
    src = Path(str(message.get("src") or ""))
    dst = Path(str(message.get("dst") or ""))
    scale = _as_int(message.get("scale"), "scale")
    quality = _as_int(message.get("quality"), "quality")
    user_tile = _as_int(message.get("tile"), "tile", 0)
    batch = _as_int(message.get("batch"), "batch", 1)
    device = str(message.get("device") or "auto")
    engine = str(message.get("engine") or "auto")
    double = bool(message.get("double"))
    image_format = str(message.get("format") or DEFAULT_FORMAT)
    lossless = bool(message.get("lossless"))
    model = _model(message)

    def progress(fraction: float, text: str) -> None:
        emit({"op": "progress", "id": job_id, "fraction": fraction, "message": text})

    # Reject a bad image before a download, and download before the GPU lock.
    preflight(src, dst, scale, image_format, quality, double, lossless)
    weight: Path | None = None
    if not uses_lanczos(model, device, engine):
        register_quietly()
        progress(0.05, "checking the model file")
        try:
            weight = ensure_weight(model, on_progress=progress)
        except StoreError as exc:
            raise ImageJobError(str(exc)) from exc
    with gpu_job(f"enlarge:{model.id}", eta_s=600.0, timeout=30.0):
        outcome = upscale_file(
            src,
            dst,
            model,
            scale=scale,
            image_format=image_format,
            quality=quality,
            double=double,
            device=device,
            user_tile=user_tile,
            batch=batch,
            engine=engine,
            on_progress=progress,
            weight=weight,
            lossless=lossless,
        )
    return _result(job_id, model.id, dst, outcome)


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """Run one message. None means the caller should stop the loop (shutdown)."""
    op = message.get("op")
    if op == "shutdown":
        return None
    if op == "ping":
        return {"op": "result", "id": str(message.get("id") or ""), "ok": True, "message": "pong", "engine": "voxprint"}
    if op == "restore_batch":
        try:
            return handle_restore(message)
        except (OSError, RuntimeError, ValueError, GpuLockTimeout) as exc:
            return {"op": "error", "id": str(message.get("id") or ""), "ok": False, "message": str(exc)}
    if op != "upscale":
        return {"op": "error", "id": str(message.get("id") or ""), "ok": False, "message": f"unknown op {op!r}"}
    try:
        return _job(message)
    except (ImageJobError, GpuLockTimeout, OSError) as exc:
        return {"op": "error", "id": str(message.get("id") or ""), "ok": False, "message": str(exc)}


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    global _stdout
    source = stdin or sys.stdin
    if stdout is not None:
        _stdout = stdout
    register_quietly()
    emit({"op": "ready", "engine": "voxprint", "pid": os.getpid()})
    for line in source:
        text = line.strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except ValueError:
            emit({"op": "error", "id": "", "ok": False, "message": "malformed json"}, stdout)
            continue
        if not isinstance(message, dict):
            emit({"op": "error", "id": "", "ok": False, "message": "message must be an object"}, stdout)
            continue
        reply = handle(message)
        if reply is None:
            emit({"op": "result", "id": str(message.get("id") or ""), "ok": True, "message": "shutdown"})
            return 0
        emit(reply)
    return 0
