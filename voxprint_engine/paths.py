"""Shared folders Clarity and Enlarge both use.

The models folder matches the other suite programs. The shared runtime folder
holds the suite Python and the refcounted manifest. The TensorRT cache does
not: engines are rebuilt from the ONNX file, so they are not refcounted and
they live under the cache directory rather than next to the weights.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_KEY = re.compile(r"[^A-Za-z0-9._+-]+")


def default_voxprint_home() -> Path:
    """The Voxprint home used when ``VOXPRINT_HOME`` is unset."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "Voxprint"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share"
    return base / "voxprint"


def voxprint_home() -> Path:
    """``VOXPRINT_HOME`` when it is set, otherwise :func:`default_voxprint_home`."""
    env = os.environ.get("VOXPRINT_HOME", "").strip()
    if env:
        return Path(env)
    return default_voxprint_home()


def voxprint_home_source() -> str:
    """``env`` when ``VOXPRINT_HOME`` is set, otherwise ``default``."""
    if os.environ.get("VOXPRINT_HOME", "").strip():
        return "env"
    return "default"


def voxprint_home_warning() -> str | None:
    """Warn when ``VOXPRINT_HOME`` hides a default home that already has a manifest.

    A stray home made rc.4 provision a second runtime beside the one the other
    suite apps were already using. The warning is empty when the variable is
    unset, when it points at the default, or when that default has no manifest yet.
    """
    if voxprint_home_source() != "env":
        return None
    chosen = voxprint_home()
    default = default_voxprint_home()
    if _same_path(chosen, default):
        return None
    manifest = default / "shared" / "manifest.json"
    if not manifest.is_file():
        return None
    return (
        f"warning: VOXPRINT_HOME is {chosen} and differs from the default {default}, "
        "which already has a shared manifest"
    )


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def suite_file() -> Path:
    """Path of the shared suite.json."""
    return voxprint_home() / "state" / "suite.json"


def shared_dir() -> Path:
    """``%LOCALAPPDATA%\\Voxprint\\shared`` or ``~/.local/share/voxprint/shared``."""
    return voxprint_home() / "shared"


def models_dir() -> Path:
    """Shared weights folder.

    ``VOXPRINT_MODELS_DIR`` wins, then ``models_dir`` in ``suite.json``, then
    ``<Voxprint home>/models`` (``%LOCALAPPDATA%\\Voxprint\\models`` on Windows,
    ``~/.local/share/voxprint/models`` on Linux).
    """
    env = os.environ.get("VOXPRINT_MODELS_DIR", "").strip().strip('"')
    if env:
        folder = Path(os.path.expandvars(os.path.expanduser(env)))
        if folder.is_absolute():
            return folder
    try:
        data = json.loads(suite_file().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = {}
    if isinstance(data, dict):
        value = data.get("models_dir")
        if isinstance(value, str) and value.strip():
            folder = Path(os.path.expandvars(os.path.expanduser(value.strip().strip('"'))))
            if folder.is_absolute():
                return folder
    return voxprint_home() / "models"


def trt_cache_root() -> Path:
    """``%LOCALAPPDATA%\\Voxprint\\trt-cache`` or ``~/.cache/voxprint/trt-cache``."""
    env = os.environ.get("VOXPRINT_TRT_CACHE", "").strip()
    if env:
        return Path(env)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "Voxprint" / "trt-cache"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".cache"
    return base / "voxprint" / "trt-cache"


def cache_key(gpu: str, driver: str, trt: str) -> str:
    """``<gpu>-<driver>-<trt>`` with characters a folder name can keep."""

    def clean(part: str) -> str:
        text = _KEY.sub("-", part.strip()).strip("-")
        return text[:80] or "unknown"

    return f"{clean(gpu)}-{clean(driver)}-{clean(trt)}"


def trt_cache_dir(gpu: str, driver: str, trt: str) -> Path:
    """One rebuildable engine folder. Nothing writes ``.users.json`` here."""
    return trt_cache_root() / cache_key(gpu, driver, trt)
