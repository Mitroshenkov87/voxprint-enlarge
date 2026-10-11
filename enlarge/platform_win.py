"""Windows specifics (guarded by ``sys.platform``): minimum build check and the Acrylic window backdrop.

Adapted from VOXPrint AI Movie Dubber ``dubber/platform_win.py`` (Apache-2.0), same approach as VOXPrint AI Audiobook Builder: DWM through ctypes only. Every function is a no-op on other platforms.

The Start-menu folder for this app is ``VOXPrint AI Suite``, shared with the other suite programs.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Optional

from enlarge import appinfo

START_MENU_FOLDER = appinfo.START_MENU_FOLDER

_WINREG: Any = None
try:
    import winreg as _imported_winreg

    _WINREG = _imported_winreg
except ImportError:  # not Windows
    _WINREG = None

IS_WINDOWS = sys.platform == "win32"
MIN_BUILD = 26100            # Windows 11 24H2
DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWA_SYSTEMBACKDROP_TYPE = 20, 33, 38
DWMSBT_TRANSIENTWINDOW = 3     # Acrylic
DWMWCP_ROUND = 2


@dataclass(frozen=True)
class OsCheck:
    """Whether this Windows build is new enough to run the app."""
    ok: bool
    build: Optional[int]
    message: str


def windows_build() -> Optional[int]:
    """The Windows build number, or 0 when this is not Windows."""
    if not IS_WINDOWS:
        return None
    try:
        getter = getattr(sys, "getwindowsversion", None)
        if getter is None:
            return None
        return int(getter().build)
    except Exception:  # noqa: BLE001
        return None


def check_os(build: Optional[int] = None, is_windows: Optional[bool] = None) -> OsCheck:
    """Soft check, never raises."""
    win = IS_WINDOWS if is_windows is None else is_windows
    if not win:
        return OsCheck(False, None, "not Windows")
    b = build if build is not None else windows_build()
    if b is None:
        return OsCheck(True, None, "")
    if b < MIN_BUILD:
        return OsCheck(False, b, f"Windows build {b} is older than {MIN_BUILD}")
    return OsCheck(True, b, "")


def windows_product_name() -> str:
    """``ProductName`` from the registry, or an empty string when it cannot be read."""
    if _WINREG is None:
        return ""
    try:
        with _WINREG.OpenKey(_WINREG.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            value, _kind = _WINREG.QueryValueEx(key, "ProductName")
    except OSError:
        return ""
    return str(value)


def backdrop_allowed(*, product: str, remote: bool, composition: bool | None) -> bool:
    """Acrylic only on a client desktop whose compositor is actually painting.

    Windows Server and a remote session report the backdrop attribute as
    success and then leave the client area black. ``composition`` False is
    the same failure. None means the query itself failed, which does not
    block a normal desktop.
    """
    if "server" in product.lower():
        return False
    if remote:
        return False
    if composition is False:
        return False
    return True


def apply_backdrop(hwnd: int, dark: bool = True) -> str:
    """Enable Acrylic through ``DwmSetWindowAttribute``. Returns ``acrylic`` or ``plain``.

    The frame is extended into the client area only after the backdrop
    attribute succeeds. A failure resets the margins to 0 and returns
    ``plain``, so the window stays opaque.
    """
    if not IS_WINDOWS:
        return "plain"
    if not backdrop_allowed(product=windows_product_name(), remote=_remote_session(), composition=_composition_enabled()):
        return "plain"
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return "plain"
        dwm = windll.dwmapi
        h = wintypes.HWND(hwnd)

        class MARGINS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_int), ("r", ctypes.c_int), ("t", ctypes.c_int), ("b", ctypes.c_int)]

        def _set(attr: int, value: int) -> int:
            v = ctypes.c_int(value)
            return dwm.DwmSetWindowAttribute(h, ctypes.c_uint(attr), ctypes.byref(v), ctypes.sizeof(v))

        def _margins(value: int) -> None:
            dwm.DwmExtendFrameIntoClientArea(h, ctypes.byref(MARGINS(value, value, value, value)))

        if _set(DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW) != 0:
            _margins(0)
            return "plain"
        _set(DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
        _set(DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
        _margins(-1)
        return "acrylic"
    except Exception:  # noqa: BLE001
        return "plain"


def _remote_session() -> bool:
    if not IS_WINDOWS:
        return False
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return False
        return bool(windll.user32.GetSystemMetrics(0x1000))
    except Exception:  # noqa: BLE001
        return False


def _composition_enabled() -> bool | None:
    """True, False, or None when DWM could not answer."""
    if not IS_WINDOWS:
        return None
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        enabled = ctypes.c_int(0)
        code = windll.dwmapi.DwmIsCompositionEnabled(ctypes.byref(enabled))
    except Exception:  # noqa: BLE001
        return None
    if code != 0:
        return None
    return bool(enabled.value)
