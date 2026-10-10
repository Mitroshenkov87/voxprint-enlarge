"""Local and shared folders.

Shared locations come from ``voxprint_engine.paths`` so Clarity and Enlarge
agree. This program's own settings stay in a separate folder so they are not
written into ``suite.json``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from voxprint_engine.paths import trt_cache_root
from voxprint_engine.paths import voxprint_home as shared_home


def voxprint_home() -> Path:
    """Shared suite root. Not created until something is saved."""
    return shared_home()


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

    A job's folder is ``<root>/<gpu>-<driver>-<trt>/``. On Windows the root is
    ``%LOCALAPPDATA%\\Voxprint\\trt-cache``. On Linux it is
    ``~/.cache/voxprint/trt-cache`` (or ``$XDG_CACHE_HOME/voxprint/trt-cache``).
    """
    return trt_cache_root()


def logs_dir() -> Path:
    """Directory for enlarge.log, created if it is missing."""
    path = app_home() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
