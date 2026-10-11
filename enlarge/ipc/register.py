"""Discovery records the installer writes.

Windows: ``HKLM`` and ``HKCU`` ``Software\\Voxprint\\Enlarge``, plus App Paths for
``voxprint-enlarge-cli.exe`` and ``voxprint-enlarge.exe``.
Linux: ``ai.voxprint.Enlarge.desktop`` plus a ``voxprint-enlarge`` launcher on ``PATH``.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import winreg
else:
    winreg = None

from enlarge.appinfo import APP_DISPLAY_NAME, product_version
from enlarge.ipc.discovery import IPC_BASE, REGISTRY_KEY
from enlarge.ipc.naming import listen_name
from enlarge.ipc.protocol import PROTOCOL

DESKTOP_NAME = "ai.voxprint.Enlarge.desktop"
LAUNCHER_NAME = "voxprint-enlarge"
APP_PATHS_DIR = r"Software\Microsoft\Windows\CurrentVersion\App Paths"
CLI_APP_PATHS = APP_PATHS_DIR + r"\voxprint-enlarge-cli.exe"
GUI_APP_PATHS = APP_PATHS_DIR + r"\voxprint-enlarge.exe"


def discovery_values(exe: Path, ipc_name: str | None = None, version: str | None = None) -> dict[str, str]:
    """Registry and desktop fields. ``version`` defaults to ``product_version()``."""
    if version is None:
        version = product_version()
    return {
        "InstallPath": str(exe.parent),
        "Exe": str(exe),
        "Version": version,
        "IpcName": ipc_name if ipc_name is not None else listen_name(),
        "IpcBase": IPC_BASE,
        "ProtocolVersion": str(PROTOCOL),
    }


def desktop_text(values: dict[str, str]) -> str:
    """The hidden Linux desktop file other apps read to find Enlarge."""
    exe = values["Exe"]
    command = f'"{exe}"' if exe.endswith(".py") else exe
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_DISPLAY_NAME}\n"
        "Comment=Offline image upscaling\n"
        f"Exec={command} --serve --hidden\n"
        "NoDisplay=true\n"
        f"X-Voxprint-Ipc={values['IpcName']}\n"
        f"X-Voxprint-Ipc-Base={values['IpcBase']}\n"
        f"X-Voxprint-Protocol={values['ProtocolVersion']}\n"
        f"X-Voxprint-Version={values['Version']}\n"
        f"X-Voxprint-Exe={exe}\n"
    )


def launcher_text(exe: Path) -> str:
    """A small PATH wrapper that execs the installed interpreter and main.py."""
    quoted = str(exe).replace("'", "'\\''")
    if exe.suffix == ".py":
        python = sys.executable.replace("'", "'\\''")
        return f"#!/bin/sh\nexec '{python}' '{quoted}' \"$@\"\n"
    return f"#!/bin/sh\nexec '{quoted}' \"$@\"\n"


def write_linux(exe: Path, data_home: Path, bin_dir: Path, ipc_name: str | None = None) -> Path:
    """Write the desktop file and a PATH launcher. Return the desktop path."""
    values = discovery_values(exe, ipc_name=ipc_name)
    apps = data_home / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    desktop = apps / DESKTOP_NAME
    desktop.write_text(desktop_text(values), encoding="utf-8")
    bin_dir.mkdir(parents=True, exist_ok=True)
    launcher = bin_dir / LAUNCHER_NAME
    launcher.write_text(launcher_text(exe), encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return desktop


def gui_exe_for(exe: Path) -> Path:
    """The window launcher beside ``exe`` when that file exists, otherwise ``exe``."""
    sibling = exe.with_name("voxprint-enlarge.exe")
    if sibling.is_file():
        return sibling
    return exe


def write_windows(exe: Path, ipc_name: str | None = None) -> None:
    """Write HKLM (best effort) and HKCU discovery values, plus both App Paths entries."""
    if winreg is None:
        raise OSError("the Windows registry is only available on Windows")
    values = discovery_values(exe, ipc_name=ipc_name)
    gui = gui_exe_for(exe)
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        _write_enlarge(hive, values)
        _write_app_path(hive, CLI_APP_PATHS, exe)
        _write_app_path(hive, GUI_APP_PATHS, gui)


def unregister_linux(data_home: Path, bin_dir: Path) -> None:
    """Remove the discovery desktop file and the PATH launcher."""
    (data_home / "applications" / DESKTOP_NAME).unlink(missing_ok=True)
    (bin_dir / LAUNCHER_NAME).unlink(missing_ok=True)


def _hive_denied(hive: int, exc: OSError) -> bool:
    """HKLM writes fail closed when the process is not elevated. HKCU does not."""
    if winreg is None or hive != winreg.HKEY_LOCAL_MACHINE:
        return False
    if isinstance(exc, PermissionError):
        return True
    return getattr(exc, "winerror", None) == 5


def _create_key(hive: int, subkey: str) -> Any | None:
    if winreg is None:
        raise OSError("the Windows registry is only available on Windows")
    try:
        return winreg.CreateKey(hive, subkey)
    except OSError as exc:
        if _hive_denied(hive, exc):
            return None
        raise


def _write_enlarge(hive: int, values: dict[str, str]) -> None:
    if winreg is None:
        raise OSError("the Windows registry is only available on Windows")
    key = _create_key(hive, REGISTRY_KEY)
    if key is None:
        return
    try:
        for name, value in values.items():
            if name == "ProtocolVersion":
                winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, int(value))
            else:
                winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    finally:
        key.Close()


def _write_app_path(hive: int, subkey: str, exe: Path) -> None:
    if winreg is None:
        raise OSError("the Windows registry is only available on Windows")
    key = _create_key(hive, subkey)
    if key is None:
        return
    try:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, str(exe))
        winreg.SetValueEx(key, "Path", 0, winreg.REG_SZ, str(exe.parent))
    finally:
        key.Close()


def unregister_windows() -> None:
    """Remove discovery from HKLM and HKCU, including both App Paths keys."""
    if winreg is None:
        raise OSError("the Windows registry is only available on Windows")
    subkeys = (REGISTRY_KEY, CLI_APP_PATHS, GUI_APP_PATHS)
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for subkey in subkeys:
            try:
                winreg.DeleteKey(hive, subkey)
            except FileNotFoundError:
                continue
            except OSError as exc:
                if getattr(exc, "winerror", None) == 2:
                    continue
                if _hive_denied(hive, exc):
                    continue
                raise


def unregister_install(data_home: Path | None = None, bin_dir: Path | None = None) -> None:
    """Drop the discovery records. The uninstaller calls this before it deletes files."""
    if sys.platform == "win32":
        unregister_windows()
        return
    if data_home is None:
        env = os.environ.get("XDG_DATA_HOME", "").strip()
        data_home = Path(env) if env else Path.home() / ".local" / "share"
    if bin_dir is None:
        bin_dir = Path.home() / ".local" / "bin"
    unregister_linux(data_home, bin_dir)


def register_install(exe: Path, data_home: Path | None = None, bin_dir: Path | None = None) -> Path | None:
    """Record this install for Lucent. Linux returns the desktop path. Windows returns None."""
    target = exe.resolve()
    if sys.platform == "win32":
        write_windows(target)
        return None
    if data_home is None:
        env = os.environ.get("XDG_DATA_HOME", "").strip()
        data_home = Path(env) if env else Path.home() / ".local" / "share"
    if bin_dir is None:
        bin_dir = Path.home() / ".local" / "bin"
    return write_linux(target, data_home, bin_dir)
