"""Where this install lives, for callers that look up Enlarge instead of the socket.

The installer writes the executable at a fixed per-user path. ``Exe`` in the
discovery record is that absolute path. The base name in APP-DISCOVERY is
``voxprint-enlarge``.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

IPC_BASE = "voxprint-enlarge"
WINDOWS_LAUNCHER = "VoxprintEnlarge.exe"
LINUX_LAUNCHER = "voxprint-enlarge"


def known_install_exe() -> Path:
    """The path the installer uses, whether or not the file is there yet."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        root = Path(local) if local else Path.home() / "AppData" / "Local"
        return root / "Voxprint" / "Enlarge" / WINDOWS_LAUNCHER
    data = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(data) if data else Path.home() / ".local" / "share"
    return root / "voxprint" / "enlarge" / LINUX_LAUNCHER


def resolve_exe(argv0: str | None = None) -> Path:
    """The running install: the known path, then ``PATH``, then this process."""
    known = known_install_exe()
    if known.is_file():
        return known.resolve()
    found = shutil.which(IPC_BASE)
    if found:
        return Path(found).resolve()
    candidate = Path(argv0 if argv0 is not None else sys.argv[0])
    if candidate.is_file():
        return candidate.resolve()
    return known
