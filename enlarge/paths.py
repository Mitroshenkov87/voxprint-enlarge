"""Local and shared folders.

Shared locations come from ``voxprint_common.paths``. Enlarge-only settings
stay in a separate folder so they are not written into ``suite.json``.
``voxprint-common`` 0.1 has ``home()`` and no label for where that path came
from, so the source string and the stray-home warning stay here. They are
part of ``--capabilities``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from voxprint_common.paths import home
from voxprint_common.paths import trt_cache_dir as shared_trt_cache


def voxprint_home() -> Path:
    """Shared suite root. Not created until something is saved."""
    return home()


def default_voxprint_home() -> Path:
    """The suite root used when ``VOXPRINT_HOME`` is unset.

    ``voxprint_common.paths.home`` does not expose this path while the
    variable is set. The stray-home warning needs both locations.
    """
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "Voxprint"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share"
    return base / "voxprint"


def voxprint_home_source() -> str:
    """``env`` when ``VOXPRINT_HOME`` is set, otherwise ``default``."""
    if os.environ.get("VOXPRINT_HOME", "").strip():
        return "env"
    return "default"


def voxprint_home_warning() -> str | None:
    """Warn when ``VOXPRINT_HOME`` hides a default home that already has a manifest."""
    if voxprint_home_source() != "env":
        return None
    chosen = voxprint_home()
    default = default_voxprint_home()
    try:
        same = chosen.resolve() == default.resolve()
    except OSError:
        same = os.path.normcase(os.path.abspath(chosen)) == os.path.normcase(os.path.abspath(default))
    if same:
        return None
    manifest = default / "shared" / "manifest.json"
    if not manifest.is_file():
        return None
    return (
        f"warning: VOXPRINT_HOME is {chosen} and differs from the default {default}, "
        "which already has a shared manifest"
    )


def app_home() -> Path:
    """Settings and logs for Enlarge only."""
    env = os.environ.get("VOXPRINT_ENLARGE_HOME", "").strip()
    if env:
        return Path(env)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "VoxprintEnlarge"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share"
    return base / "voxprint-enlarge"


def default_models_dir() -> Path:
    """``<Voxprint home>/models``, the suite models folder."""
    return voxprint_home() / "models"


def trt_cache_dir() -> Path:
    """Root of the rebuildable TensorRT cache. Engines here are not refcounted.

    ``VOXPRINT_TRT_CACHE`` wins. Otherwise the cache is
    ``%LOCALAPPDATA%\\Voxprint\\trt-cache`` on Windows and
    ``~/.cache/voxprint/trt-cache`` on Linux. It does not follow ``VOXPRINT_HOME``.
    """
    return shared_trt_cache()


def logs_dir() -> Path:
    """Directory for enlarge.log, created if it is missing."""
    path = app_home() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
