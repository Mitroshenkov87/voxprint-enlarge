"""Output containers. AVIF is the default. PNG is an input, never an output.

The main set is AVIF, JPEG XL (lossy or lossless), and WebP. JPEG is legacy.
HEIC is legacy and only listed when :func:`heic_backend` finds a system codec.
Encoders we ship are permissively licensed: Pillow's AVIF (libavif), JPEG, and
WebP, and libjxl through ``imagecodecs`` (BSD-3-Clause). No separate AVIF
plugin and no HEIC encoder are dependencies.
"""
from __future__ import annotations

from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image

from voxprint_engine.errors import ImageJobError
from voxprint_engine.heic_system import detect_backend, encode_with_heif_enc, encode_with_libheif

DEFAULT_FORMAT = "avif"
DEFAULT_QUALITY = 60
OUTPUT_IDS = ("avif", "jxl", "webp", "jpeg", "heic")
LEGACY_IDS = frozenset({"jpeg", "heic"})
_EXTENSIONS = {"avif": ".avif", "jxl": ".jxl", "webp": ".webp", "jpeg": ".jpg", "heic": ".heic"}
_LABELS = {
    "avif": "AVIF",
    "jxl": "JPEG XL",
    "webp": "WebP",
    "jpeg": "JPEG (legacy)",
    "heic": "HEIC (legacy)",
}

_heic_cache: str | None = None
_heic_known = False


def _avif_ready() -> bool:
    """Pillow 11.3 and newer ship an AVIF saver. No extra plugin is imported."""
    Image.init()
    return "AVIF" in Image.SAVE


def heic_backend() -> str | None:
    """Which system HEIC encoder is present, if any. Cached for the process."""
    global _heic_cache, _heic_known
    if not _heic_known:
        _heic_cache = detect_backend()
        _heic_known = True
    return _heic_cache


def reset_heic_cache() -> None:
    """Tests call this before replacing detection."""
    global _heic_cache, _heic_known
    _heic_cache = None
    _heic_known = False


def choices() -> list[tuple[str, str]]:
    """``(label, id)`` for the Output step and Settings. HEIC is last, and only when present."""
    rows = [(label, key) for key, label in _LABELS.items() if key != "heic"]
    if heic_backend() is not None:
        rows.append((_LABELS["heic"], "heic"))
    return rows


def normalize_format(value: object) -> str:
    """A stored format id, or the AVIF default when the value is missing or no longer allowed."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "jpg":
            key = "jpeg"
        if key in OUTPUT_IDS and (key != "heic" or heic_backend() is not None):
            return key
    return DEFAULT_FORMAT


def output_extension(image_format: str) -> str:
    key = require_output_format(image_format, lossless=False)
    return _EXTENSIONS[key]


def require_output_format(image_format: str, *, lossless: bool) -> str:
    """Reject PNG and anything outside the output set. HEIC also needs a system codec."""
    key = image_format.strip().lower()
    if key == "jpg":
        key = "jpeg"
    if key == "png":
        raise ImageJobError("PNG is an input format. Save as AVIF, JPEG XL, or WebP.")
    if key not in OUTPUT_IDS:
        raise ImageJobError("format must be avif, jxl, webp, jpeg, or heic")
    if key == "heic" and heic_backend() is None:
        raise ImageJobError("HEIC needs a system codec. This app does not bundle one.")
    if lossless and key != "jxl":
        raise ImageJobError("lossless output is JPEG XL only")
    return key


def _jxl_distance(quality: int) -> float:
    """Map the 1–100 slider onto libjxl distance. 100 is 0, 80 is 1, 60 is 2."""
    return max(0.0, (100 - int(quality)) / 20.0)


def open_rgb(path: Path) -> Image.Image:
    """Decode one still image to RGB. JPEG XL is read with imagecodecs; the rest with Pillow."""
    if path.suffix.lower() == ".jxl":
        array = np.asarray(imagecodecs.jpegxl_decode(path.read_bytes()))
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB").copy()
    with Image.open(path) as image:
        return image.convert("RGB")


def save_image(frame: Image.Image, dest: Path, image_format: str, quality: int, lossless: bool = False) -> None:
    """Write ``frame`` in an allowed output format."""
    key = require_output_format(image_format, lossless=lossless)
    if quality < 1 or quality > 100:
        raise ImageJobError("quality must be from 1 to 100")
    image = frame.convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if key == "avif":
        if not _avif_ready():
            raise ImageJobError("the AVIF encoder is not available")
        image.save(dest, format="AVIF", quality=quality, speed=6)
        return
    if key == "jxl":
        array = np.ascontiguousarray(np.asarray(image))
        if lossless:
            blob = imagecodecs.jpegxl_encode(array, lossless=True)
        else:
            blob = imagecodecs.jpegxl_encode(array, distance=_jxl_distance(quality))
        dest.write_bytes(blob)
        return
    if key == "webp":
        image.save(dest, format="WEBP", quality=quality, method=6)
        return
    if key == "jpeg":
        image.save(dest, format="JPEG", quality=quality, subsampling=1)
        return
    backend = heic_backend()
    if backend == "pillow":
        Image.init()
        container = "HEIF" if "HEIF" in Image.SAVE else "HEIC"
        image.save(dest, format=container, quality=quality)
        return
    if backend == "libheif":
        encode_with_libheif(image, dest, quality)
        return
    if backend == "heif-enc":
        encode_with_heif_enc(image, dest, quality)
        return
    raise ImageJobError("HEIC needs a system codec. This app does not bundle one.")
