"""Put CUDA 13 and TensorRT 10 on the library path before ONNX Runtime loads them.

ONNX Runtime 1.31's CUDA provider needs ``cublasLt64_13.dll`` (and the rest of
the CUDA 13 runtime) from the ``nvidia-*-cu13`` wheels. Its TensorRT provider
opens ``nvinfer_10`` from ``tensorrt-cu13-libs`` 10.14.1.48. Those directories
are resolved in this process. An inherited ``PATH`` is not enough: a worker
started without a console does not see them unless this module adds them.

``os.add_dll_directory`` handles are kept alive for the life of the process.
Child environments get the same directories prepended to ``PATH`` or
``LD_LIBRARY_PATH``. On Linux, glibc reads ``LD_LIBRARY_PATH`` only at process
start, so a process that was not given those directories re-executes once.
Forcing the CUDA wheels in with ``RTLD_GLOBAL`` crashes torch in the same
process, which is what the audit hit.
"""
from __future__ import annotations

import importlib.metadata as metadata
import os
import sys
import sysconfig
from collections.abc import MutableMapping
from pathlib import Path

TRT_LIBS_VERSION = "10.14.1.48"
TRT_LIBS_PACKAGE = "tensorrt-cu13-libs"
_LIB_NAMES = ("nvinfer_10.dll", "libnvinfer.so.10")
_EXTRA_LIBS = ("libnvinfer_plugin.so.10", "libnvonnxparser.so.10", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll")
_ORDER = (
    "nvjitlink",
    "cuda_runtime",
    "cuda_nvrtc",
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvtx",
    "nvinfer",
    "tensorrt",
)

_searched = False
_folder: Path | None = None
_dirs_ready = False
_dirs: list[Path] = []
_dll_handles: list[object] = []
_added: set[str] = set()
_prepared = False


def library_dir() -> Path | None:
    """Directory that contains ``nvinfer_10``, or ``None`` when it is not installed."""
    global _searched, _folder
    if _searched:
        return _folder
    _searched = True
    _folder = _find_library_dir()
    return _folder


def native_library_dirs() -> list[Path]:
    """CUDA 13 wheel ``bin``/``lib`` directories, then the TensorRT 10 library directory."""
    global _dirs_ready, _dirs
    if _dirs_ready:
        return list(_dirs)
    found = _cuda_wheel_dirs()
    trt = library_dir()
    if trt is not None:
        found.append(trt)
    _dirs = _unique(_ordered(found))
    _dirs_ready = True
    return list(_dirs)


def prepare_native_runtime() -> list[Path]:
    """Register CUDA 13 and TensorRT in this process, before ``onnxruntime`` is imported.

    Safe to call more than once. Returns the directories that were found.
    """
    global _prepared
    folders = native_library_dirs()
    if _prepared:
        return folders
    if sys.platform == "win32":
        _register_windows(folders)
    _prepend(os.environ, folders)
    _prepared = True
    _reexec_for_linux(folders)
    return folders


def register_tensorrt_libraries() -> Path | None:
    """Load TensorRT 10 and the CUDA 13 wheels. Safe to call more than once."""
    prepare_native_runtime()
    return library_dir()


def apply_library_path(env: dict[str, str]) -> None:
    """Prepend the CUDA 13 and TensorRT directories to a child environment."""
    _prepend(env, native_library_dirs())


def _register_windows(folders: list[Path]) -> None:
    """Keep ``os.add_dll_directory`` handles so the CUDA provider can open them."""
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return
    for folder in folders:
        key = str(folder)
        if key in _added:
            continue
        try:
            _dll_handles.append(add(key))
        except OSError:
            continue
        _added.add(key)


def _reexec_for_linux(folders: list[Path]) -> None:
    """Restart this process once so glibc sees the CUDA 13 library directories.

    Pytest and the CI stand-in skip this. ``RTLD_GLOBAL`` is not used: loading
    the CUDA wheels that way crashes torch later in the same process.
    """
    if not _should_reexec(folders):
        return
    os.environ["VOXPRINT_NATIVE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _should_reexec(folders: list[Path]) -> bool:
    if sys.platform == "win32" or not folders:
        return False
    if os.environ.get("VOXPRINT_NATIVE_REEXEC") == "1":
        return False
    if os.environ.get("VOXPRINT_CI_LANCZOS") or "pytest" in sys.modules:
        return False
    return True


def _prepend(env: MutableMapping[str, str], folders: list[Path]) -> None:
    if not folders:
        return
    key = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    current = env.get(key, "")
    parts = [part for part in current.split(os.pathsep) if part]
    prefix = [str(folder) for folder in folders if str(folder) not in parts]
    if not prefix:
        return
    env[key] = os.pathsep.join([*prefix, *parts])


def _cuda_wheel_dirs() -> list[Path]:
    found: list[Path] = []
    roots = _site_roots()
    for dist in _distributions():
        name = _dist_name(dist)
        if not name.startswith("nvidia-") or "cu13" not in name:
            continue
        located = _locate_root(dist)
        if located is not None:
            roots.append(located)
    seen: set[Path] = set()
    for root in roots:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        try:
            children = list(nvidia.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            for sub in ("bin", "lib", Path("lib") / "x64"):
                folder = child / sub
                try:
                    resolved = folder.resolve()
                except OSError:
                    continue
                if resolved in seen or not _has_cu13(folder):
                    continue
                seen.add(resolved)
                found.append(folder)
    return found


def _site_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            roots.append(Path(value))
    return roots


def _distributions() -> list[object]:
    try:
        return list(metadata.distributions())
    except (OSError, ValueError):
        return []


def _dist_name(dist: object) -> str:
    meta = getattr(dist, "metadata", None)
    raw = meta.get("Name") if meta is not None else None
    return raw.strip().lower() if isinstance(raw, str) else ""


def _locate_root(dist: object) -> Path | None:
    locate = getattr(dist, "locate_file", None)
    if locate is None:
        return None
    try:
        return Path(str(locate("")))
    except (OSError, TypeError, ValueError):
        return None


def _has_cu13(folder: Path) -> bool:
    try:
        names = [path.name.lower() for path in folder.iterdir() if path.is_file()]
    except OSError:
        return False
    return any("_13" in name or ".so.13" in name or "cu13" in name for name in names)


def _ordered(folders: list[Path]) -> list[Path]:
    def rank(folder: Path) -> tuple[int, str]:
        text = str(folder).lower().replace("\\", "/")
        for index, token in enumerate(_ORDER):
            if token in text:
                return index, text
        return len(_ORDER), text

    return sorted(folders, key=rank)


def _unique(folders: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for folder in folders:
        try:
            resolved = folder.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(folder)
    return result


def _find_library_dir() -> Path | None:
    roots: list[Path] = []
    try:
        dist = metadata.distribution(TRT_LIBS_PACKAGE)
    except metadata.PackageNotFoundError:
        dist = None
    if dist is not None:
        roots.append(Path(str(dist.locate_file(""))))
    roots.extend(_site_roots())
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
