"""Worker handler for ``restore_batch``. Additive. Enlarge's image ops stay as they are.

This file is part of the patch against voxprint-enlarge 6d00d44. It imports
the engine; Clarity does not copy it into the application package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from voxprint_common.gpu_lock import gpu_job
from voxprint_engine.providers import ProviderChoice, choices_for, cuda_fallback_choices
from voxprint_engine.registry.schema import checked_in_channels
from voxprint_engine.session import EngineBuildError, available_providers, open_session
from voxprint_engine.video import restore_frames, run_batch
from voxprint_engine.vram import is_memory_error


def _restore_with_fallback(
    attempts: list[list[ProviderChoice]],
    onnx: Path,
    frames: np.ndarray,
    *,
    tile: int,
    overlap: int,
    scale: int,
    batch: int,
    length: int,
    window_overlap: int,
) -> tuple[np.ndarray, str]:
    """TensorRT, then CUDA when the engine does not build or the kernel fails."""
    errors: list[str] = []
    for choices in attempts:
        if not choices:
            continue
        try:
            session, provider = open_session(onnx, choices)
        except EngineBuildError as exc:
            errors.append(str(exc))
            continue

        def infer(batch_nchw: np.ndarray, _session: Any = session) -> np.ndarray:
            return run_batch(_session, batch_nchw)

        try:
            restored = restore_frames(
                frames,
                infer,
                tile=tile,
                overlap=overlap,
                scale=scale,
                batch=batch,
                length=length,
                window_overlap=window_overlap,
            )
        except RuntimeError as exc:
            if is_memory_error(exc):
                raise
            errors.append(str(exc))
            continue
        finally:
            closer = getattr(session, "close", None)
            if callable(closer):
                closer()
        return restored, provider
    raise RuntimeError("; ".join(errors) or "the ONNX session did not load")


def handle_restore(message: dict[str, Any]) -> dict[str, Any]:
    """Load a NHWC float ``.npy``, restore it, and write the output ``.npy``.

    The GPU lock covers the session build and the restore. TensorRT is opened
    first. CUDA is the next attempt when that build fails.
    """
    job_id = str(message.get("id") or "")
    frames_path = Path(str(message.get("frames_npy") or ""))
    out_path = Path(str(message.get("out_npy") or ""))
    onnx = Path(str(message.get("onnx") or ""))
    if not frames_path.is_file():
        raise FileNotFoundError("frames_npy was not found")
    if not onnx.is_file():
        raise FileNotFoundError("the ONNX file was not found")
    frames = np.load(frames_path, allow_pickle=False)
    if frames.ndim != 4:
        raise ValueError("frames_npy must be NHWC")
    tile = int(message.get("tile") or 512)
    overlap = int(message.get("overlap") or 16)
    batch = int(message.get("batch") or 1)
    scale = int(message.get("scale") or 1)
    length = int(message.get("length") or 1)
    window_overlap = int(message.get("window_overlap") or 0)
    device = str(message.get("device") or "auto")
    owner = str(message.get("owner") or "clarity")
    raw_channels = message.get("in_channels", 3)
    if raw_channels is None:
        raw_channels = 3
    channels = checked_in_channels(raw_channels)
    installed = available_providers()
    attempts = [choices_for(installed, device, in_channels=channels)]
    fallback = cuda_fallback_choices(installed, device, in_channels=channels)
    if fallback:
        attempts.append(fallback)

    with gpu_job(owner, "restore", eta=600.0, timeout=30.0):
        restored, provider = _restore_with_fallback(
            attempts,
            onnx,
            frames.astype(np.float32, copy=False),
            tile=tile,
            overlap=overlap,
            scale=scale,
            batch=batch,
            length=length,
            window_overlap=window_overlap,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, restored)
    return {
        "op": "result",
        "id": job_id,
        "ok": True,
        "message": f"onnx {provider}",
        "dst": str(out_path),
        "engine": "onnx",
        "frames": int(restored.shape[0]),
    }
