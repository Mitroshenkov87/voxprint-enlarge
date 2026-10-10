"""``--capabilities --json``. This command exits 0 with or without a GPU."""
from __future__ import annotations

from typing import Any

from enlarge.appinfo import APP_VERSION
from enlarge.ipc.protocol import PROTOCOL
from voxprint_engine.formats import DEFAULT_FORMAT, heic_backend
from voxprint_engine.hardware import MIN_DRIVER_MAJOR, capability_warning, driver_major, query_gpus
from voxprint_engine.worker.lanczos import IMAGE_EXTENSIONS

PHOTO_FAST = "4x-nomos2-realplksr-dysample"
PHOTO_MAX = "4x-nomos2-hq-dat2"
ART = "realesrgan-x4plus-anime6b"
SCALES = [2, 4]
_INPUT_NAMES = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".webp": "webp",
    ".avif": "avif",
    ".jxl": "jxl",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
}


def gpu_capability() -> dict[str, str]:
    """``status`` is ``ok``, ``warn``, or ``none``. ``reason`` is present only when ``none``."""
    gpus = query_gpus()
    if not gpus:
        return {"status": "none", "reason": "no_nvidia"}
    gpu = gpus[0]
    major = driver_major(gpu.driver)
    if major is None or major < MIN_DRIVER_MAJOR:
        return {"status": "none", "reason": "driver_too_old"}
    if capability_warning(gpu.compute):
        return {"status": "warn"}
    return {"status": "ok"}


def model_catalog() -> list[dict[str, Any]]:
    """Presets shipped today. The preset enum is open: later presets can be added."""
    rows = [
        ("photo", "fast", PHOTO_FAST),
        ("photo", "max", PHOTO_MAX),
        ("art", "fast", ART),
        ("art", "max", ART),
    ]
    return [{"id": model_id, "preset": preset, "quality": quality, "scales": list(SCALES)} for preset, quality, model_id in rows]


def input_formats() -> list[str]:
    names = {_INPUT_NAMES[suffix] for suffix in IMAGE_EXTENSIONS if suffix in _INPUT_NAMES}
    return sorted(names)


def output_formats() -> list[str]:
    """AVIF, JPEG XL, WebP, and JPEG. HEIC only when a system codec is present. PNG is never an output."""
    formats = ["avif", "jxl", "webp", "jpeg"]
    if heic_backend() is not None:
        formats.append("heic")
    return formats


def capabilities() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "version": APP_VERSION,
        "gpu": gpu_capability(),
        "models": model_catalog(),
        "input_formats": input_formats(),
        "output_formats": output_formats(),
        "default_output_format": DEFAULT_FORMAT,
    }
