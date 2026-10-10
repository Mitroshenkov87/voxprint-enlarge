"""Worker main loop. Reads jobs from stdin, writes one JSON line per event to stdout."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from voxprint_engine.gpu_lock import GpuLockTimeout, gpu_job
from voxprint_engine.vram import choose_tile, measure, note_spill, overlap_for
from voxprint_engine.worker.lanczos import ImageJobError, lanczos_resize


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


def _job(message: dict[str, Any]) -> dict[str, Any]:
    job_id = str(message.get("id") or "")
    src = Path(str(message.get("src") or ""))
    dst = Path(str(message.get("dst") or ""))
    scale = _as_int(message.get("scale"), "scale")
    quality = _as_int(message.get("quality"), "quality")
    user_tile = _as_int(message.get("tile"), "tile", 0)
    model_tile = _as_int(message.get("model_tile"), "model_tile", 0)
    batch = _as_int(message.get("batch"), "batch", 1)
    device = str(message.get("device") or "auto")
    model_id = str(message.get("model_id") or "")
    double = bool(message.get("double"))
    image_format = str(message.get("format") or "png")
    snapshot = measure(device)
    tile = choose_tile(snapshot.budget_gb, model_tile, user_tile)
    overlap = overlap_for(tile)
    emit({
        "op": "progress",
        "id": job_id,
        "fraction": 0.05,
        "message": f"tile {tile}, overlap {overlap}, VRAM budget {snapshot.budget_gb:.1f} GB",
    })
    # The resize does not use the GPU yet. The lock is still taken so the suite
    # sees this job, and so the real engine can replace the body without a new protocol.
    with gpu_job(f"enlarge:{model_id or 'preview'}", eta_s=120.0, timeout=30.0):
        emit({"op": "progress", "id": job_id, "fraction": 0.4, "message": "resizing"})
        width, height = lanczos_resize(src, dst, scale, image_format, quality, double)
    spilled = False
    return {
        "op": "result",
        "id": job_id,
        "ok": True,
        "message": "lanczos preview",
        "dst": str(dst),
        "width": width,
        "height": height,
        "scale": scale * scale if double else scale,
        "engine": "lanczos",
        "model_id": model_id,
        "tile": tile,
        "overlap": overlap,
        "vram_budget_gb": round(snapshot.budget_gb, 3),
        "spilled": spilled,
        "next_batch": note_spill(batch, spilled),
    }


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """Run one message. None means the caller should stop the loop (shutdown)."""
    op = message.get("op")
    if op == "shutdown":
        return None
    if op == "ping":
        return {"op": "result", "id": str(message.get("id") or ""), "ok": True, "message": "pong", "engine": "lanczos"}
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
    emit({"op": "ready", "engine": "lanczos", "pid": os.getpid()})
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
