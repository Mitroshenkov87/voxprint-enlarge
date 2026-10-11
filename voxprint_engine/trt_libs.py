"""Put TensorRT 10 on the library path before ONNX Runtime loads it.

ONNX Runtime 1.31's TensorRT execution provider opens ``nvinfer_10``. The
unpinned ``tensorrt-cu13`` package currently installs TensorRT 11
(``nvinfer_11``), so the provider never loads. The GPU installer pins
``tensorrt-cu13-libs`` 10.14.1.48. This module does not import ONNX Runtime.
"""
from __future__ import annotations

import ctypes
import importlib.metadata as metadata
import os
import sys
import sysconfig
from pathlib import Path

TRT_LIBS_VERSION = "10.14.1.48"
TRT_LIBS_PACKAGE = "tensorrt-cu13-libs"
_LIB_NAMES = ("nvinfer_10.dll", "libnvinfer.so.10")
_EXTRA_LIBS = ("libnvinfer_plugin.so.10", "libnvonnxparser.so.10", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll")

_searched = False
_folder: Path | None = None
_registered = False


def library_dir() -> Path | None:
    """Directory that contains ``nvinfer_10``, or ``None`` when it is not installed."""
    global _searched, _folder
    if _searched:
        return _folder
    _searched = True
    _folder = _find_library_dir()
    return _folder


def register_tensorrt_libraries() -> Path | None:
    """Load TensorRT 10 into this process. Safe to call more than once.

    Windows uses ``os.add_dll_directory``. Linux loads the shared objects with
    ``RTLD_GLOBAL`` because glibc only reads ``LD_LIBRARY_PATH`` at process start.
    """
    global _registered
    folder = library_dir()
    if folder is None:
        return None
    if _registered:
        return folder
    if sys.platform == "win32":
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            add(str(folder))
    else:
        for name in ("libnvinfer.so.10", *_EXTRA_LIBS):
            path = folder / name
            if path.is_file() and path.suffix != ".dll":
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _registered = True
    return folder


def apply_library_path(env: dict[str, str]) -> None:
    """Prepend the TensorRT library directory to the child process search path."""
    folder = library_dir()
    if folder is None:
        return
    key = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    current = env.get(key, "")
    text = str(folder)
    parts = [part for part in current.split(os.pathsep) if part]
    if text in parts:
        return
    env[key] = text + (os.pathsep + current if current else "")


def _find_library_dir() -> Path | None:
    roots: list[Path] = []
    try:
        dist = metadata.distribution(TRT_LIBS_PACKAGE)
    except metadata.PackageNotFoundError:
        dist = None
    if dist is not None:
        roots.append(Path(str(dist.locate_file(""))))
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            roots.append(Path(value))
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        for name in _LIB_NAMES:
            match = _first_named(resolved, name)
            if match is not None:
                return match.parent
    return None


def _first_named(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    nested = root / "tensorrt_libs" / name
    if nested.is_file():
        return nested
    # One directory level. A full rglob of site-packages walks the torch tree
    # and delays every worker start.
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        candidate = child / name
        if candidate.is_file():
            return candidate
        nested_child = child / "tensorrt_libs" / name
        if nested_child.is_file():
            return nested_child
    return None
