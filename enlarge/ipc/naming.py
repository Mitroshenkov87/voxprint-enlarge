"""Socket location, pipe name, and output paths that never overwrite a file."""
from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path


def pipe_name(user_sid: str) -> str:
    """``voxprint-enlarge-`` plus the first 16 hex chars of ``sha256(sid)``."""
    digest = hashlib.sha256(user_sid.encode("utf-8")).hexdigest()[:16]
    return f"voxprint-enlarge-{digest}"


def current_user_sid() -> str:
    """Windows user SID. Other platforms return an empty string."""
    if sys.platform != "win32":
        return ""
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        return ""

    class _SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    class _TokenUser(ctypes.Structure):
        _fields_ = [("User", _SidAndAttributes)]

    try:
        length = wintypes.DWORD(0)
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(length))
        buffer = ctypes.create_string_buffer(max(length.value, 1))
        if not advapi32.GetTokenInformation(token, 1, buffer, length, ctypes.byref(length)):
            return ""
        user = ctypes.cast(buffer, ctypes.POINTER(_TokenUser)).contents
        text = ctypes.c_wchar_p()
        if not advapi32.ConvertSidToStringSidW(user.User.Sid, ctypes.byref(text)):
            return ""
        try:
            return text.value or ""
        finally:
            kernel32.LocalFree(text)
    finally:
        kernel32.CloseHandle(token)


def runtime_root() -> Path:
    """``$XDG_RUNTIME_DIR``, or a private directory under the temp folder."""
    env = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if env:
        return Path(env)
    if sys.platform != "win32":
        return Path(tempfile.gettempdir()) / f"voxprint-run-{os.getuid()}"
    return Path(tempfile.gettempdir()) / "voxprint-run"


def socket_path() -> Path:
    """Linux socket: ``$XDG_RUNTIME_DIR/voxprint/enlarge.sock``."""
    return runtime_root() / "voxprint" / "enlarge.sock"


def listen_name() -> str:
    """Name passed to ``QLocalServer.listen``."""
    if sys.platform == "win32":
        return pipe_name(current_user_sid() or os.environ.get("USERNAME", "user"))
    return str(socket_path())


def prepare_endpoint() -> str:
    """Create the socket directory at mode 0700 on Linux. Return the listen name."""
    if sys.platform == "win32":
        return listen_name()
    path = socket_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if path.exists():
        path.unlink()
    return str(path)


def tighten_socket(name: str) -> None:
    """Linux socket mode 0600. Windows ACLs come from ``QLocalServer.UserAccessOption``."""
    if sys.platform == "win32":
        return
    path = Path(name)
    if path.exists():
        os.chmod(path, 0o600)


def _reserve(candidates: list[Path]) -> Path:
    last = candidates[0]
    for candidate in candidates:
        last = candidate
        candidate.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise OSError(f"could not reserve an output path near {last}")


def _default_names(source: Path, scale: int, extension: str, directory: Path | None) -> list[Path]:
    folder = directory or source.parent
    stem = source.stem
    names = [folder / f"{stem}.enlarged-{scale}x{extension}"]
    names.extend(folder / f"{stem}.enlarged-{scale}x-{index}{extension}" for index in range(2, 1000))
    return names


def _explicit_names(path: Path) -> list[Path]:
    names = [path]
    names.extend(path.with_name(f"{path.stem}-{index}{path.suffix}") for index in range(2, 1000))
    return names


def _unwritable(path: Path) -> bool:
    parent = path.parent
    return parent.exists() and not parent.is_dir()


def peek_default(source: Path, scale: int, extension: str, directory: Path | None = None) -> Path:
    """The path ``reserve_default`` would create, without creating it."""
    for candidate in _default_names(source, scale, extension, directory):
        if _unwritable(candidate):
            raise OSError(f"could not reserve an output path near {candidate}")
        if not candidate.exists():
            return candidate
    raise OSError(f"could not reserve an output path near {source}")


def peek_explicit(path: Path) -> Path:
    """The path ``reserve_explicit`` would create, without creating it."""
    for candidate in _explicit_names(path):
        if _unwritable(candidate):
            raise OSError(f"could not reserve an output path near {candidate}")
        if not candidate.exists():
            return candidate
    raise OSError(f"could not reserve an output path near {path}")


def reserve_default(source: Path, scale: int, extension: str, directory: Path | None = None) -> Path:
    """``<name>.enlarged-<scale>x.<ext>``, then ``-2``, ``-3``, without replacing a file."""
    return _reserve(_default_names(source, scale, extension, directory))


def reserve_explicit(path: Path) -> Path:
    """Use ``path`` when it is free, otherwise ``<stem>-2<suffix>`` and so on."""
    return _reserve(_explicit_names(path))
