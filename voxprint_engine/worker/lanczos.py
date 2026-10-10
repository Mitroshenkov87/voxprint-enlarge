"""Resize used by the CI stand-in and by a 4x network that must land on 2x.

``VOXPRINT_CI_LANCZOS=1`` is the only way a whole job uses this resize.
``VOXPRINT_FAKE_DELAY`` (seconds, capped) exists so tests can observe the GPU
lock while a job is in flight.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image

from voxprint_engine.errors import ImageJobError
from voxprint_engine.formats import open_rgb, require_output_format, save_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".m2ts", ".mpg", ".mpeg"}
MAX_OUTPUT_PIXELS = 80_000_000
MAX_DELAY_S = 5.0


def output_scale(scale: int, double: bool) -> int:
    """Pixel scale after the optional second pass. The model itself stays 4x."""
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


def lanczos_resize(
    src: Path,
    dst: Path,
    scale: int,
    image_format: str,
    quality: int,
    double: bool = False,
    lossless: bool = False,
) -> tuple[int, int]:
    """Write ``src`` resized by ``scale`` (twice when ``double``). Returns the new size."""
    _delay()
    suffix = src.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        raise ImageJobError("Enlarge takes images only. Video belongs to Voxprint AI Clarity.")
    if suffix not in IMAGE_EXTENSIONS:
        raise ImageJobError(f"unsupported image type {suffix or '(none)'}")
    require_output_format(image_format, lossless=lossless)
    if quality < 1 or quality > 100:
        raise ImageJobError("quality must be from 1 to 100")
    factor = output_scale(scale, double)
    if not src.is_file():
        raise ImageJobError("source image was not found")
    if src.resolve() == dst.resolve():
        raise ImageJobError("output path must be different from the source")
    try:
        frame = open_rgb(src)
        width, height = frame.size
        out_w, out_h = width * factor, height * factor
        if out_w < 1 or out_h < 1 or out_w * out_h > MAX_OUTPUT_PIXELS:
            raise ImageJobError("output would be empty or larger than 80 million pixels")
        resized = frame.resize((out_w, out_h), Image.Resampling.LANCZOS)
        save_image(resized, dst, image_format, quality, lossless)
    except ImageJobError:
        raise
    except (OSError, ValueError) as exc:
        raise ImageJobError(f"could not resize the image: {exc}") from exc
    return out_w, out_h
