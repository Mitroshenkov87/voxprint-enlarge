"""Discovery records a future installer writes. There is no installer in this build.

Windows: ``HKCU\\Software\\Voxprint\\Enlarge`` and App Paths.
Linux: ``ai.voxprint.Enlarge.desktop`` plus a ``voxprint-enlarge`` launcher on ``PATH``.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

if sys.platform == "win32":
    import winreg
else:
    winreg = None

from enlarge.appinfo import APP_DISPLAY_NAME, APP_VERSION
from enlarge.ipc.naming import listen_name
from enlarge.ipc.protocol import PROTOCOL

DESKTOP_NAME = "ai.voxprint.Enlarge.desktop"
LAUNCHER_NAME = "voxprint-enlarge"
REGISTRY_KEY = r"Software\Voxprint\Enlarge"
APP_PATHS_KEY = r"Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-enlarge.exe"


def discovery_values(exe: Path, ipc_name: str | None = None, version: str = APP_VERSION) -> dict[str, str]:
    return {
        "InstallPath": str(exe.parent),
        "Exe": str(exe),
        "Version": version,
        "IpcName": ipc_name if ipc_name is not None else listen_name(),
        "ProtocolVersion": str(PROTOCOL),
    }


def desktop_text(values: dict[str, str]) -> str:
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
        f"X-Voxprint-Protocol={values['ProtocolVersion']}\n"
    )


def launcher_text(exe: Path) -> str:
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


def write_windows(exe: Path, ipc_name: str | None = None) -> None:
    """Write HKCU discovery values and the per-user App Paths entry."""
    if winreg is None:
        raise OSError("the Windows registry is only available on Windows")
    values = discovery_values(exe, ipc_name=ipc_name)
    enlarge_key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY)
    try:
        for name, value in values.items():
            winreg.SetValueEx(enlarge_key, name, 0, winreg.REG_SZ, value)
    finally:
        enlarge_key.Close()
    app_key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, APP_PATHS_KEY)
    try:
        winreg.SetValueEx(app_key, "", 0, winreg.REG_SZ, values["Exe"])
        winreg.SetValueEx(app_key, "Path", 0, winreg.REG_SZ, values["InstallPath"])
    finally:
        app_key.Close()


def unregister_linux(data_home: Path, bin_dir: Path) -> None:
    """Remove the discovery desktop file and the PATH launcher."""
    (data_home / "applications" / DESKTOP_NAME).unlink(missing_ok=True)
    (bin_dir / LAUNCHER_NAME).unlink(missing_ok=True)


def unregister_windows() -> None:
    """Remove the HKCU discovery key and the App Paths entry."""
    if winreg is None:
        raise OSError("the Windows registry is only available on Windows")
    for key in (REGISTRY_KEY, APP_PATHS_KEY):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
        except FileNotFoundError:
            continue
        except OSError as exc:
            if getattr(exc, "winerror", None) == 2:
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
