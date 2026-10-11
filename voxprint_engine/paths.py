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


def voxprint_home() -> Path:
    """``%LOCALAPPDATA%\\Voxprint`` or ``~/.local/share/voxprint``."""
    env = os.environ.get("VOXPRINT_HOME", "").strip()
    if env:
        return Path(env)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "Voxprint"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share"
    return base / "voxprint"


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
