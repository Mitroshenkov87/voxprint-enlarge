"""``--capabilities --json``. This command exits 0 with or without a GPU."""
from __future__ import annotations

from typing import Any

from enlarge.appinfo import product_version
from enlarge.ipc.protocol import PROTOCOL
from enlarge.paths import voxprint_home, voxprint_home_source
from voxprint_common.gpu_probe import MIN_DRIVER_MAJOR, capability_warning, driver_major
from voxprint_engine.nvidia_smi import query_gpus
from voxprint_engine.formats import DEFAULT_FORMAT, heic_backend
from voxprint_engine.registry import load_builtin
from voxprint_engine.store import weight_report
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


def _cc(compute: tuple[int, int] | None) -> str | None:
    if compute is None:
        return None
    return f"{compute[0]}.{compute[1]}"


def gpu_capability() -> dict[str, Any]:
    """``status`` is ``ok``, ``warn``, or ``none``. ``reason`` is null unless ``status`` is ``none``."""
    gpus = query_gpus()
    if not gpus:
        return {"name": None, "cc": None, "status": "none", "reason": "no_nvidia"}
    gpu = gpus[0]
    cc = _cc(gpu.compute)
    major = driver_major(gpu.driver)
    if major is None or major < MIN_DRIVER_MAJOR:
        return {"name": gpu.name, "cc": cc, "status": "none", "reason": "driver_too_old"}
    if capability_warning("enlarge", gpu.compute):
        return {"name": gpu.name, "cc": cc, "status": "warn", "reason": None}
    return {"name": gpu.name, "cc": cc, "status": "ok", "reason": None}


def model_catalog() -> list[dict[str, Any]]:
    """Presets shipped today. ``installed`` is true only when the pinned file is on disk."""
    catalog = load_builtin(release=True)
    rows = [
        ("photo", "fast", PHOTO_FAST),
        ("photo", "max", PHOTO_MAX),
        ("art", "fast", ART),
        ("art", "max", ART),
    ]
    listed: list[dict[str, Any]] = []
    for preset, quality, model_id in rows:
        model = catalog.get(model_id)
        installed = model is not None and weight_report(model)[0] == "ready"
        listed.append(
            {
                "id": model_id,
                "preset": preset,
                "quality": quality,
                "scales": list(SCALES),
                "installed": installed,
            }
        )
    return listed


def input_formats() -> list[str]:
    """Sorted input format names the worker can open."""
    names = {_INPUT_NAMES[suffix] for suffix in IMAGE_EXTENSIONS if suffix in _INPUT_NAMES}
    return sorted(names)


def output_formats() -> list[str]:
    """AVIF, JPEG XL, WebP, and JPEG. HEIC only when a system codec is present. PNG is never an output."""
    formats = ["avif", "jxl", "webp", "jpeg"]
    if heic_backend() is not None:
        formats.append("heic")
    return formats


def capabilities() -> dict[str, Any]:
    """One capabilities object. ``version`` is ``product_version()``."""
    return {
        "protocol": PROTOCOL,
        "app": "enlarge",
        "version": product_version(),
        "gpu": gpu_capability(),
        "models": model_catalog(),
        "presets": ["photo", "art"],
        "max_scale": 4,
        "input_formats": input_formats(),
        "output_formats": output_formats(),
        "default_output_format": DEFAULT_FORMAT,
        "voxprint_home": str(voxprint_home()),
        "voxprint_home_source": voxprint_home_source(),
    }
