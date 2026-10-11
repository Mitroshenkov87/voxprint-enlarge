"""Turn a pinned ``.pth`` into a checked ONNX file and cache it by sha256.

Torch 2.14's default dynamo exporter writes a graph that ``onnx.checker``
rejects ("two nodes with same node name"). The legacy exporter
(``dynamo=False``) is the path that checks out. The cached file is fp16.

A manifest may also name a prebuilt ``.onnx`` (``onnx_url`` and
``onnx_sha256``). That file is preferred and skips conversion. The loader is
spandrel's MIT package. ``spandrel_extra_arches`` is never imported.
"""
from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from voxprint_engine.paths import models_dir
from voxprint_engine.registry.schema import ModelManifest, checked_in_channels
from voxprint_engine.store import StoreError, download, ensure_weight, sha256_file

# These imports stay inside the export functions. The worker imports this
# module on every job, and importing torch at module level takes long enough
# on Windows that the GPU lock is released before a test can see it.
_torch: Any = None
_onnx: Any = None
_to_fp16: Any = None
_loader: Any = None

Progress = Callable[[float, str], None]
Downloader = Callable[[str, Path, str, Progress | None], None]
Converter = Callable[[Path, Path], None]


class OnnxExportError(StoreError):
    """The weight could not be turned into a checked ONNX file."""


def onnx_cache_path(digest: str, root: Path | None = None) -> Path:
    """``<models>/<sha256>.onnx``. ``digest`` is 64 lowercase hex characters."""
    text = digest.strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise OnnxExportError("the ONNX cache name must be a sha256")
    return (root or models_dir()) / f"{text}.onnx"


def cached_onnx(model: ModelManifest, root: Path | None = None) -> Path | None:
    """The cached ONNX file when it is already on disk, else ``None``.

    A prebuilt source is stored under ``onnx_sha256``. A conversion is stored
    under the weight ``sha256``.
    """
    digest = model.onnx_sha256 if model.onnx_url and model.onnx_sha256 else model.sha256
    if not digest:
        return None
    try:
        path = onnx_cache_path(digest, root)
    except OnnxExportError:
        return None
    if _cache_ready(path):
        return path
    return None


def export_module(module: Any, dest: Path, *, height: int = 64, width: int = 64, channels: int = 3) -> None:
    """Write a checked fp16 ONNX file for an eval-mode torch module.

    The example tensor is ``1 x channels x height x width``. ``channels``
    defaults to 3. A 4-channel network such as DRUNet needs the noise-level
    map in that count, or the exported graph cannot feed TensorRT. The legacy
    exporter is requested explicitly. A torch build that does not accept
    ``dynamo`` is retried once without that argument, and the result still
    has to pass ``onnx.checker``.
    """
    try:
        channels = checked_in_channels(channels)
    except ValueError as exc:
        raise OnnxExportError(str(exc)) from exc
    if height < 1 or width < 1:
        raise OnnxExportError("the export size must be at least 1 pixel")
    torch = _load_torch()
    onnx = _load_onnx()
    to_fp16 = _load_fp16()
    module.eval()
    module = module.cpu().float()
    example = torch.rand(1, channels, height, width)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    checked = dest.with_name(dest.name + ".checked")
    try:
        _legacy_export(torch, module, example, partial)
        graph = onnx.load(str(partial))
        graph = to_fp16(graph, keep_io_types=False)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(checked))
        os.replace(checked, dest)
    except OnnxExportError:
        raise
    except Exception as exc:
        raise OnnxExportError(f"ONNX export failed: {exc}") from exc
    finally:
        partial.unlink(missing_ok=True)
        checked.unlink(missing_ok=True)


def export_weight(weight: Path, dest: Path, on_progress: Progress | None = None, *, channels: int = 3) -> None:
    """Load ``weight`` with spandrel's MIT core and write a checked fp16 ONNX file.

    ``channels`` is the example tensor's depth. The manifest field
    ``in_channels`` is what :func:`ensure_onnx` passes.
    """
    if not weight.is_file():
        raise OnnxExportError(f"not a file: {weight.name}")
    loader_cls = _load_spandrel()
    _emit(on_progress, 0.2, "loading the weight")
    try:
        descriptor = loader_cls().load_from_file(str(weight))
    except OnnxExportError:
        raise
    except Exception as exc:
        raise OnnxExportError(f"could not load {weight.name}: {exc}") from exc
    _emit(on_progress, 0.5, f"converting {weight.name} to ONNX")
    export_module(descriptor.model, dest, channels=channels)
    _emit(on_progress, 1.0, "checked the ONNX file")


def ensure_onnx(
    model: ModelManifest,
    root: Path | None = None,
    on_progress: Progress | None = None,
    *,
    weight: Path | None = None,
    converter: Converter | None = None,
    downloader: Downloader | None = None,
) -> Path:
    """Return a checked ONNX file for ``model``.

    When the manifest lists ``onnx_url`` and ``onnx_sha256``, that file is
    downloaded into the sha256 cache and conversion is skipped. Otherwise the
    pinned ``.pth`` is converted once and stored as ``<weight sha256>.onnx``.
    A later call returns the cached file.
    """
    folder = root or models_dir()
    fetch = downloader or download
    if model.onnx_url and model.onnx_sha256:
        return _prebuilt(model, folder, on_progress, fetch)
    if not model.sha256:
        raise OnnxExportError(f"{model.id} has no pinned weight to convert")
    dest = onnx_cache_path(model.sha256, folder)
    if _cache_ready(dest):
        _emit(on_progress, 1.0, "ONNX file is cached")
        return dest
    if weight is None:
        weight = ensure_weight(model, folder, on_progress)
    elif not weight.is_file():
        raise OnnxExportError(f"{model.id} weight is missing")
    sibling = weight.with_suffix(".onnx")
    if sibling.is_file() and sibling.resolve() != dest.resolve():
        adopted = _adopt_sibling(sibling, dest)
        if adopted is not None:
            _emit(on_progress, 1.0, "ONNX file is cached")
            return adopted
    _emit(on_progress, 0.4, f"converting {weight.name} to ONNX")
    channels = model.in_channels
    convert = converter or (lambda src, target: export_weight(src, target, on_progress, channels=channels))
    try:
        convert(weight, dest)
    except OnnxExportError:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise OnnxExportError(f"ONNX export failed: {exc}") from exc
    if not _cache_ready(dest):
        raise OnnxExportError(f"{model.id} conversion did not write an ONNX file")
    try:
        _check_saved(dest)
    except OnnxExportError:
        dest.unlink(missing_ok=True)
        raise
    _emit(on_progress, 1.0, "checked the ONNX file")
    return dest


def _prebuilt(model: ModelManifest, folder: Path, on_progress: Progress | None, fetch: Downloader) -> Path:
    dest = onnx_cache_path(model.onnx_sha256, folder)
    expected = model.onnx_sha256.strip().lower()
    if _cache_ready(dest) and sha256_file(dest) == expected:
        _emit(on_progress, 1.0, "ONNX file is cached")
        return dest
    _emit(on_progress, 0.1, f"downloading {dest.name}")
    fetch(model.onnx_url, dest, expected, on_progress)
    if sha256_file(dest) != expected:
        dest.unlink(missing_ok=True)
        raise OnnxExportError(f"sha256 mismatch for {dest.name}")
    try:
        _check_saved(dest)
    except OnnxExportError:
        dest.unlink(missing_ok=True)
        raise
    _emit(on_progress, 1.0, "checked the ONNX file")
    return dest


def _adopt_sibling(sibling: Path, dest: Path) -> Path | None:
    """Copy a sibling ``.onnx`` into the cache when the checker accepts it."""
    try:
        _check_saved(sibling)
    except OnnxExportError:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    try:
        shutil.copyfile(sibling, partial)
        os.replace(partial, dest)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise OnnxExportError(f"could not cache {sibling.name}: {exc}") from exc
    return dest


def _check_saved(path: Path) -> None:
    onnx = _load_onnx()
    try:
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
    except OnnxExportError:
        raise
    except Exception as exc:
        raise OnnxExportError(f"the ONNX file failed the checker: {exc}") from exc


def _cache_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _emit(on_progress: Progress | None, fraction: float, message: str) -> None:
    if on_progress is not None:
        on_progress(fraction, message)


def _load_torch() -> Any:
    global _torch
    if _torch is None:
        try:
            import torch
        except ImportError as exc:
            raise OnnxExportError("export needs torch. See docs/ENGINE.md.") from exc
        _torch = torch
    return _torch


def _load_onnx() -> Any:
    global _onnx
    if _onnx is None:
        try:
            import onnx
        except ImportError as exc:
            raise OnnxExportError("export needs onnx. See docs/ENGINE.md.") from exc
        _onnx = onnx
    return _onnx


def _load_fp16() -> Any:
    global _to_fp16
    if _to_fp16 is None:
        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16
        except ImportError as exc:
            raise OnnxExportError("export needs onnxruntime. See docs/ENGINE.md.") from exc
        _to_fp16 = convert_float_to_float16
    return _to_fp16


def _load_spandrel() -> Any:
    """The MIT ``spandrel`` loader. ``spandrel_extra_arches`` is never imported."""
    global _loader
    if _loader is None:
        try:
            from spandrel import ModelLoader
        except ImportError as exc:
            raise OnnxExportError("export needs spandrel. See docs/ENGINE.md.") from exc
        _loader = ModelLoader
    return _loader


def _legacy_export(torch_mod: Any, module: Any, example: Any, dest: Path) -> None:
    """Export with ``dynamo=False``. Retry without the argument on older torch."""
    kwargs: dict[str, Any] = {
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {2: "height", 3: "width"},
            "output": {2: "height_out", 3: "width_out"},
        },
        "opset_version": 17,
        "dynamo": False,
    }
    try:
        _export_legacy(torch_mod, module, example, dest, kwargs)
    except TypeError:
        kwargs.pop("dynamo")
        _export_legacy(torch_mod, module, example, dest, kwargs)


def _export_legacy(torch_mod: Any, module: Any, example: Any, dest: Path, kwargs: dict[str, Any]) -> None:
    """Call the legacy exporter. Its deprecation warning is the dynamo default we are avoiding."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch_mod.onnx.export(module, example, str(dest), **kwargs)
