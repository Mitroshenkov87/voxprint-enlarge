"""HEIC only through a codec the operating system already has.

Nothing in this module bundles libheif, x265, or pillow-heif. Detection looks
for a HEIF saver Pillow already has, a system libheif that can encode HEVC, or
a system ``heif-enc`` binary. If none of those exist, HEIC stays hidden.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HEVC = 1
_COLOR_RGB = 1
_CHROMA_RGB = 10
_CHANNEL_INTERLEAVED = 10


class _HeifError(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("subcode", ctypes.c_int), ("message", ctypes.c_char_p)]


def pillow_heif_registered() -> bool:
    """True when some other module already gave Pillow a HEIF saver. We never register one."""
    Image.init()
    names = set(getattr(Image, "SAVE", {}))
    return "HEIF" in names or "HEIC" in names


def _library() -> ctypes.CDLL | None:
    found = ctypes.util.find_library("heif")
    if not found:
        return None
    try:
        return ctypes.CDLL(found)
    except OSError:
        return None


def _bind(lib: ctypes.CDLL) -> bool:
    try:
        lib.heif_init.argtypes = [ctypes.c_void_p]
        lib.heif_init.restype = _HeifError
        lib.heif_have_encoder_for_format.argtypes = [ctypes.c_int]
        lib.heif_have_encoder_for_format.restype = ctypes.c_int
        lib.heif_context_alloc.argtypes = []
        lib.heif_context_alloc.restype = ctypes.c_void_p
        lib.heif_context_free.argtypes = [ctypes.c_void_p]
        lib.heif_context_free.restype = None
        lib.heif_context_get_encoder_for_format.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.heif_context_get_encoder_for_format.restype = _HeifError
        lib.heif_encoder_set_lossy_quality.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.heif_encoder_set_lossy_quality.restype = _HeifError
        lib.heif_encoder_release.argtypes = [ctypes.c_void_p]
        lib.heif_encoder_release.restype = None
        lib.heif_image_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.heif_image_create.restype = _HeifError
        lib.heif_image_add_plane.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.heif_image_add_plane.restype = _HeifError
        lib.heif_image_get_plane.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.heif_image_get_plane.restype = ctypes.POINTER(ctypes.c_uint8)
        lib.heif_image_release.argtypes = [ctypes.c_void_p]
        lib.heif_image_release.restype = None
        lib.heif_context_encode_image.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.heif_context_encode_image.restype = _HeifError
        lib.heif_context_write_to_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.heif_context_write_to_file.restype = _HeifError
    except AttributeError:
        return False
    return True


_inited = False


def _ensure_init(lib: ctypes.CDLL) -> bool:
    global _inited
    if _inited:
        return True
    error = lib.heif_init(None)
    if error.code != 0:
        return False
    _inited = True
    return True


def libheif_can_encode() -> bool:
    """True when the system libheif has an HEVC encoder plugin. Decode-only installs are false."""
    lib = _library()
    if lib is None or not _bind(lib) or not _ensure_init(lib):
        return False
    try:
        return bool(lib.heif_have_encoder_for_format(HEVC))
    except (OSError, AttributeError):
        return False


def _check(error: _HeifError, action: str) -> None:
    if error.code != 0:
        detail = error.message.decode("utf-8", "replace") if error.message else action
        raise OSError(detail)


def encode_with_libheif(frame: Image.Image, dest: Path, quality: int) -> None:
    """Write one RGB frame with the system libheif HEVC encoder."""
    lib = _library()
    if lib is None or not _bind(lib) or not _ensure_init(lib):
        raise OSError("the system libheif encoder is not available")
    image = frame.convert("RGB")
    width, height = image.size
    pixels = image.tobytes()
    context = lib.heif_context_alloc()
    encoder = ctypes.c_void_p()
    heif_image = ctypes.c_void_p()
    if not context:
        raise OSError("libheif could not open a context")
    try:
        _check(lib.heif_context_get_encoder_for_format(context, HEVC, ctypes.byref(encoder)), "encoder")
        _check(lib.heif_encoder_set_lossy_quality(encoder, int(quality)), "quality")
        _check(
            lib.heif_image_create(width, height, _COLOR_RGB, _CHROMA_RGB, ctypes.byref(heif_image)),
            "image",
        )
        _check(lib.heif_image_add_plane(heif_image, _CHANNEL_INTERLEAVED, width, height, 8), "plane")
        stride = ctypes.c_int()
        plane = lib.heif_image_get_plane(heif_image, _CHANNEL_INTERLEAVED, ctypes.byref(stride))
        if not plane or stride.value < width * 3:
            raise OSError("libheif did not allocate an RGB plane")
        row = width * 3
        for y in range(height):
            ctypes.memmove(ctypes.addressof(plane[y * stride.value]), pixels[y * row:(y + 1) * row], row)
        _check(lib.heif_context_encode_image(context, heif_image, encoder, None, None), "encode")
        dest.parent.mkdir(parents=True, exist_ok=True)
        _check(lib.heif_context_write_to_file(context, os.fsencode(dest)), "write")
    finally:
        if heif_image:
            lib.heif_image_release(heif_image)
        if encoder:
            lib.heif_encoder_release(encoder)
        lib.heif_context_free(context)


def heif_enc_path() -> str | None:
    """Path of a system heif-enc binary, or None when this machine has none."""
    return shutil.which("heif-enc")


def heif_enc_can_encode() -> bool:
    """True when the system ``heif-enc`` binary lists an HEVC encoder."""
    exe = heif_enc_path()
    if not exe:
        return False
    try:
        if os.name == "nt":
            proc = subprocess.run(
                [exe, "--list-encoders"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        else:
            proc = subprocess.run([exe, "--list-encoders"], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    text = f"{proc.stdout}\n{proc.stderr}".upper()
    return any(token in text for token in ("HEVC", "H.265", "H265", "X265", "KVAZAAR"))


def encode_with_heif_enc(frame: Image.Image, dest: Path, quality: int) -> None:
    """Write one HEIC file with the system encoder. The app does not bundle one."""
    exe = heif_enc_path()
    if not exe:
        raise OSError("heif-enc is not installed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voxprint-heic-") as tmp:
        source = Path(tmp) / "frame.png"
        frame.convert("RGB").save(source, format="PNG")
        command = [exe, "-q", str(int(quality)), "-o", str(dest), str(source)]
        try:
            if os.name == "nt":
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                )
            else:
                proc = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise OSError(f"heif-enc failed: {exc}") from exc
    if proc.returncode != 0 or not dest.is_file():
        detail = (proc.stderr or proc.stdout or "heif-enc failed").strip()
        raise OSError(detail)


def detect_backend() -> str | None:
    """``pillow``, ``libheif``, ``heif-enc``, or None. The result is not cached here."""
    if pillow_heif_registered():
        return "pillow"
    if libheif_can_encode():
        return "libheif"
    if heif_enc_can_encode():
        return "heif-enc"
    return None
