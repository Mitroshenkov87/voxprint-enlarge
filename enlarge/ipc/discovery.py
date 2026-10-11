"""Where this install lives, for callers that look up Enlarge instead of the socket.

The installer writes the CLI at the suite path. ``Exe`` in the discovery
record is that absolute path. The base name in APP-DISCOVERY is
``voxprint-enlarge``. Readers still accept an older per-user install for one
version when the new file is absent.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    import winreg
else:
    winreg = None

IPC_BASE = "voxprint-enlarge"
CLI_LAUNCHER = "voxprint-enlarge-cli.exe"
GUI_LAUNCHER = "voxprint-enlarge.exe"
LEGACY_WINDOWS_LAUNCHER = "VoxprintEnlarge.exe"
LINUX_LAUNCHER = "voxprint-enlarge"
REGISTRY_KEY = r"Software\Voxprint\Enlarge"
_LINUX_INSTALL = Path("/opt/voxprint-ai-suite/enlarge")
_WHICH_NAMES = ("voxprint-enlarge-cli", "voxprint-enlarge")


def known_install_exe() -> Path:
    """The path the current installer uses, whether or not the file is there yet."""
    if sys.platform == "win32":
        return _windows_program_files() / "VOXPrint AI Suite" / "Enlarge" / CLI_LAUNCHER
    return _LINUX_INSTALL / LINUX_LAUNCHER


def legacy_install_exe() -> Path:
    """The per-user path published before the suite folder. One version still reads it."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        root = Path(local) if local else Path.home() / "AppData" / "Local"
        return root / "Voxprint" / "Enlarge" / LEGACY_WINDOWS_LAUNCHER
    data = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(data) if data else Path.home() / ".local" / "share"
    return root / "voxprint" / "enlarge" / LINUX_LAUNCHER


def resolve_exe(argv0: str | None = None) -> Path:
    """Suite path, then the registry, then the legacy path, then ``PATH``, then this process."""
    known = known_install_exe()
    if known.is_file():
        return known.resolve()
    recorded = _registry_exe()
    if recorded is not None and recorded.is_file():
        return recorded.resolve()
    legacy = legacy_install_exe()
    if legacy.is_file():
        return legacy.resolve()
    for name in _WHICH_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    candidate = Path(argv0 if argv0 is not None else sys.argv[0])
    if candidate.is_file():
        return candidate.resolve()
    return known


def _windows_program_files() -> Path:
    for key in ("ProgramW6432", "ProgramFiles"):
        value = os.environ.get(key, "").strip()
        if value:
            return Path(value)
    return Path("C:/Program Files")


def _registry_exe() -> Path | None:
    """``Exe`` from HKLM, then HKCU. An older per-user value is still accepted."""
    if winreg is None:
        return None
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            key = winreg.OpenKey(hive, REGISTRY_KEY)
        except OSError:
            continue
        try:
            raw, _kind = winreg.QueryValueEx(key, "Exe")
        except OSError:
            raw = ""
        finally:
            key.Close()
        text = str(raw).strip()
        if text:
            return Path(text)
    return None
