"""Stand-in upscaler: a high-quality Lanczos resize, images only.

The real engine will swap this for ONNX Runtime. Callers pass the same request
either way. ``VOXPRINT_FAKE_DELAY`` (seconds, capped) exists so tests can observe
the GPU lock while a job is in flight.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts", ".mpg", ".mpeg"}
FORMATS = {"png": "PNG", "jpeg": "JPEG", "jpg": "JPEG", "webp": "WEBP"}
MAX_OUTPUT_PIXELS = 80_000_000
MAX_DELAY_S = 5.0


class ImageJobError(ValueError):
    """The request is not an image job this worker can finish."""


def output_scale(scale: int, double: bool) -> int:
    if scale not in (2, 4):
        raise ImageJobError("scale must be 2 or 4")
    factor = scale * scale if double else scale
    if factor > 16:
        raise ImageJobError("scale is too large")
    return factor


def _delay() -> None:
    raw = os.environ.get("VOXPRINT_FAKE_DELAY", "").strip()
    if not raw:
        return
    try:
        seconds = float(raw)
    except ValueError:
        return
    if seconds > 0:
        time.sleep(min(seconds, MAX_DELAY_S))


def lanczos_resize(src: Path, dst: Path, scale: int, image_format: str, quality: int, double: bool = False) -> tuple[int, int]:
    """Write ``src`` resized by ``scale`` (twice when ``double``). Returns the new size."""
    _delay()
    suffix = src.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        raise ImageJobError("Enlarge takes images only. Video belongs to Voxprint AI Clarity.")
    if suffix not in IMAGE_EXTENSIONS:
        raise ImageJobError(f"unsupported image type {suffix or '(none)'}")
    fmt = FORMATS.get(image_format.strip().lower())
    if fmt is None:
        raise ImageJobError("format must be png, jpeg, or webp")
    if quality < 1 or quality > 100:
        raise ImageJobError("quality must be from 1 to 100")
    factor = output_scale(scale, double)
    if not src.is_file():
        raise ImageJobError("source image was not found")
    if src.resolve() == dst.resolve():
        raise ImageJobError("output path must be different from the source")
    try:
        with Image.open(src) as image:
            frame = image.convert("RGB")
            width, height = frame.size
            out_w, out_h = width * factor, height * factor
            if out_w < 1 or out_h < 1 or out_w * out_h > MAX_OUTPUT_PIXELS:
                raise ImageJobError("output would be empty or larger than 80 million pixels")
            resized = frame.resize((out_w, out_h), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs: dict[str, object] = {}
            if fmt == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["subsampling"] = 1
            elif fmt == "WEBP":
                save_kwargs["quality"] = quality
            resized.save(dst, format=fmt, **save_kwargs)
    except ImageJobError:
        raise
    except (OSError, ValueError) as exc:
        raise ImageJobError(f"could not resize the image: {exc}") from exc
    return out_w, out_h
