"""Local and shared folders.

Shared Voxprint home matches the other suite apps: ``%LOCALAPPDATA%\\Voxprint`` on
Windows and ``~/.local/share/voxprint`` on Linux (or ``$XDG_DATA_HOME/voxprint``).
This program's own settings stay in a separate folder so they are not written
into ``suite.json``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def voxprint_home() -> Path:
    """Shared suite root. Not created until something is saved."""
    env = os.environ.get("VOXPRINT_HOME", "").strip()
    if env:
        return Path(env)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "Voxprint"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share"
    return base / "voxprint"


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
    """Future TensorRT cache. Not created in this build.

    Clarity's layout is ``<home>/trt-cache/<gpu>-<driver>-<trt>/``. The pieces
    of that name are added when an engine is actually built.
    """
    return voxprint_home() / "trt-cache"


def logs_dir() -> Path:
    path = app_home() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
