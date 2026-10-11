"""Shared suite runtime: one folder, one refcounted manifest.

Enlarge and Clarity both call ``ensure_runtime`` and ``release_runtime``.
The folder is ``%LOCALAPPDATA%\\Voxprint\\shared`` on Windows and
``~/.local/share/voxprint/shared`` elsewhere. ``manifest.json`` lists each
component, its version, and the apps that hold it. A payload is removed
only when that list is empty.

``VOXPRINT_CI_LANCZOS=1`` skips the network and reports every component as
skipped, so tests stay offline. Pass ``allow_skip=False`` to force the check.
"""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from voxprint_engine.hardware import ci_lanczos
from voxprint_engine.onnx_export import OnnxExportError, cached_onnx, ensure_onnx
from voxprint_engine.paths import models_dir, shared_dir, voxprint_home, voxprint_home_source, voxprint_home_warning
from voxprint_engine.registry import DEFAULT_MODEL_ID, ModelManifest, load_builtin
from voxprint_engine.shared_manifest import (
    ORT_LOCK,
    PYTHON_LOCK,
    TRT_LOCK,
    TORCH_LOCK,
    SharedManifest,
    SharedManifestError,
    find_reusable,
    load,
    lock_hash,
    remove_ref,
    reserve,
    update,
)
from voxprint_engine.store import StoreError, register, unregister, weight_report

PYTHON_VERSION = "3.14"
GPU_VERSION = "cu130"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
ORT_GPU_SPEC = "onnxruntime-gpu==1.31.0"
TRT_LIBS_SPEC = "tensorrt-cu13-libs==10.14.1.48"
# Same pins as the ``gpu`` extra in pyproject.toml. ensure_runtime always installs them.
GPU_EXTRA = (ORT_GPU_SPEC, TRT_LIBS_SPEC)
TRT_META_SPEC = "tensorrt-cu13==10.14.1.48"
NVIDIA_INDEX = "https://pypi.nvidia.com"
SPANDREL_SPEC = "spandrel>=0.4,<0.5"
ONNX_SPEC = "onnx>=1.16,<2"

ComponentId = Literal["python", "gpu", "model"]
Progress = Callable[[str], None]
Probe = Callable[[Path], bool]
Installer = Callable[[Path, Progress], None]
Remover = Callable[[Path], None]

_APP_ID = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_GPU_PROBE = """
import importlib.metadata as meta
import importlib.util as util
import sys
missing = [name for name in ("torch", "onnxruntime", "spandrel", "onnx") if util.find_spec(name) is None]
names = {
    dist.metadata["Name"].lower()
    for dist in meta.distributions()
    if dist.metadata and dist.metadata.get("Name")
}
if "onnxruntime" in names:
    missing.append("cpu-onnxruntime")
try:
    libs = meta.version("tensorrt-cu13-libs")
except meta.PackageNotFoundError:
    missing.append("tensorrt-cu13-libs")
else:
    if not str(libs).startswith("10.14.1.48"):
        missing.append("tensorrt-cu13-libs")
try:
    meta_version = meta.version("tensorrt-cu13")
except meta.PackageNotFoundError:
    missing.append("tensorrt-cu13")
else:
    if not str(meta_version).startswith("10.14.1.48"):
        missing.append("tensorrt-cu13")
for banned in ("tensorrt", "tensorrt-cu13-bindings"):
    try:
        meta.version(banned)
    except meta.PackageNotFoundError:
        continue
    missing.append(banned)
sys.exit(0 if not missing else 1)
"""


class SuiteInstallError(RuntimeError):
    """The shared runtime could not be checked, installed, or released."""


@dataclass(frozen=True)
class ComponentStatus:
    """One shared component after a check, a repair, or a release."""

    id: str
    version: str
    state: str
    holders: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict[str, object]:
        """JSON-ready fields."""
        return {
            "detail": self.detail,
            "holders": list(self.holders),
            "id": self.id,
            "state": self.state,
            "version": self.version,
        }


@dataclass(frozen=True)
class RuntimeReport:
    """Result of ``ensure_runtime`` or ``release_runtime``."""

    ok: bool
    skipped: bool
    components: tuple[ComponentStatus, ...]

    def as_dict(self) -> dict[str, object]:
        """JSON-ready fields. ``--selftest --json`` prints this object."""
        return {
            "components": [item.as_dict() for item in self.components],
            "ok": self.ok,
            "skipped": self.skipped,
        }


@dataclass(frozen=True)
class _Pin:
    id: str
    kind: str
    version: str
    path: str
    sha256: str
    unit: ComponentId


def _pins() -> tuple[_Pin, ...]:
    digest = _default_model().sha256.lower()
    prefix = digest[:12]
    return (
        _Pin("python-3.14", "runtime", PYTHON_VERSION, "python", lock_hash(PYTHON_LOCK), "python"),
        _Pin("torch-cu130", "wheels", GPU_VERSION, "python", lock_hash(TORCH_LOCK), "gpu"),
        _Pin("onnxruntime-gpu-1.31", "wheels", "1.31.0", "python", lock_hash(ORT_LOCK), "gpu"),
        _Pin("tensorrt-10", "wheels", "10.14.1.48", "python", lock_hash(TRT_LOCK), "gpu"),
        _Pin(f"model-{prefix}", "model", DEFAULT_MODEL_ID, f"model/{prefix}", digest, "model"),
    )


def managed_ids() -> tuple[str, ...]:
    """Component ids ``ensure_runtime`` records in the shared manifest."""
    return tuple(pin.id for pin in _pins())


def read_manifest(root: Path | None = None) -> dict[str, object]:
    """Return the suite manifest. A missing or damaged file is an empty schema 1 document.

    An old Canvas or Enlarge file is migrated to schema 1 before it is returned.
    """
    try:
        return load(root).as_dict()
    except SharedManifestError as exc:
        raise SuiteInstallError(str(exc)) from exc


def ensure_runtime(
    app: str,
    *,
    repair: bool = True,
    progress: Progress | None = None,
    root: Path | None = None,
    probes: Mapping[str, Probe] | None = None,
    installers: Mapping[str, Installer] | None = None,
    allow_skip: bool = True,
) -> RuntimeReport:
    """Hold ``app`` on each shared component, installing anything that is missing.

    The same app a second time does not add a second ref. ``probes`` and
    ``installers`` replace the real checks for tests. The real steps copy or
    record Python 3.14, install the CUDA 13 wheels (torch cu130, ONNX Runtime
    GPU, TensorRT 10), and download the default model. A component another app
    refs is left in place and the new copy is recorded beside it.
    """
    _check_app(app)
    _emit(progress, f"Voxprint home: {voxprint_home()} ({voxprint_home_source()})")
    warning = voxprint_home_warning()
    if warning:
        _emit(progress, warning)
    if allow_skip and ci_lanczos():
        return _skipped_report()
    folder = root or shared_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pins = _pins()
    try:
        snapshot = load(folder)
    except SharedManifestError as exc:
        raise SuiteInstallError(str(exc)) from exc
    assignments = _assign(snapshot, app, pins)
    outcomes: dict[str, tuple[str, str]] = {}
    units: tuple[ComponentId, ...] = ("python", "gpu", "model")
    for unit in units:
        rel = _unit_path(assignments, unit)
        for pin, component_id, _path in assignments:
            if pin.unit == unit:
                _emit(progress, f"Checking {component_id} {pin.version}")
        if _unit_ready(snapshot, assignments, unit, folder, probes):
            outcomes[unit] = ("ok", "ready")
            continue
        if not repair:
            outcomes[unit] = ("missing", "not installed")
            continue
        for pin, component_id, _path in assignments:
            if pin.unit == unit:
                _emit(progress, f"Installing {component_id} {pin.version}")
        try:
            _call_installer(unit, folder, progress, installers, rel)
        except (SuiteInstallError, StoreError, OSError, SharedManifestError) as exc:
            outcomes[unit] = ("failed", str(exc))
            continue
        if _call_probe(unit, folder, probes, rel):
            outcomes[unit] = ("repaired", "repaired")
        else:
            outcomes[unit] = ("failed", "still missing after install")
    records: dict[str, dict[str, object]] = {}
    if any(state in {"ok", "repaired"} for state, _detail in outcomes.values()):
        try:
            with update(folder) as manifest:
                for pin, component_id, path in assignments:
                    state, _detail = outcomes[pin.unit]
                    if state not in {"ok", "repaired"}:
                        continue
                    record, _reused = reserve(
                        manifest,
                        component_id=component_id,
                        kind=pin.kind,
                        version=pin.version,
                        path=path,
                        sha256=pin.sha256,
                        app=app,
                    )
                    records[component_id] = record
        except SharedManifestError as exc:
            raise SuiteInstallError(str(exc)) from exc
    statuses: list[ComponentStatus] = []
    for pin, component_id, _path in assignments:
        state, detail = outcomes[pin.unit]
        holders = tuple(_entry_refs(records.get(component_id)))
        if state == "ok":
            _emit(progress, f"{component_id} {pin.version} is ready")
        if pin.unit == "model" and state in {"ok", "repaired"}:
            _register_model_user(app)
        statuses.append(ComponentStatus(component_id, pin.version, state, holders, detail))
    ok = all(item.state in {"ok", "repaired"} for item in statuses)
    return RuntimeReport(ok, False, tuple(statuses))


def release_runtime(
    app: str,
    *,
    progress: Progress | None = None,
    root: Path | None = None,
    removers: Mapping[str, Remover] | None = None,
) -> RuntimeReport:
    """Drop ``app`` from the manifest. Delete a payload only when its ref list is empty."""
    _check_app(app)
    folder = root or shared_dir()
    pins = _pins()
    known = {pin.id for pin in pins}
    if not (folder / "manifest.json").is_file():
        return RuntimeReport(
            True,
            False,
            tuple(ComponentStatus(pin.id, pin.version, "ok", (), "absent") for pin in pins),
        )
    failed = False
    removed: set[str] = set()
    failed_detail: dict[str, str] = {}
    try:
        with update(folder) as manifest:
            pending: list[tuple[_Pin, str, dict[str, object]]] = []
            for pin in pins:
                entry = manifest.components.get(pin.id)
                if not isinstance(entry, dict) or app not in _entry_refs(entry):
                    continue
                others = [name for name in _entry_refs(entry) if name != app]
                if others:
                    remove_ref(manifest, pin.id, app)
                    if pin.unit == "model":
                        _unregister_quietly(app)
                    _emit(progress, f"{pin.id} {pin.version} still used by {', '.join(others)}")
                    continue
                saved = dict(entry)
                saved["refs"] = list(_entry_refs(entry))
                rel = str(entry.get("path") or pin.path)
                remove_ref(manifest, pin.id, app)
                del manifest.components[pin.id]
                pending.append((pin, rel, saved))
            extras: list[tuple[str, str, dict[str, object]]] = []
            for key, entry in list(manifest.components.items()):
                if key in known or not isinstance(entry, dict) or app not in _entry_refs(entry):
                    continue
                others = [name for name in _entry_refs(entry) if name != app]
                if others:
                    remove_ref(manifest, key, app)
                    continue
                saved = dict(entry)
                saved["refs"] = list(_entry_refs(entry))
                rel = str(entry.get("path") or "")
                remove_ref(manifest, key, app)
                del manifest.components[key]
                extras.append((key, rel, saved))
            by_unit: dict[str, list[tuple[_Pin, str, dict[str, object]]]] = {"gpu": [], "python": [], "model": []}
            for row in pending:
                by_unit[row[0].unit].append(row)
            for unit in ("gpu", "python", "model"):
                rows = by_unit[unit]
                if not rows:
                    continue
                rels: list[str] = []
                for _pin, rel, _saved in rows:
                    if rel not in rels:
                        rels.append(rel)
                try:
                    for rel in rels:
                        if _path_used(manifest, rel):
                            continue
                        _emit(progress, f"Removing {unit}")
                        _call_remover(unit, folder, app, removers, rel)
                except (SuiteInstallError, StoreError, OSError) as exc:
                    failed = True
                    failed_detail[unit] = str(exc)
                    for pin, _rel, saved in rows:
                        manifest.components[pin.id] = saved
                else:
                    for pin, _rel, _saved in rows:
                        removed.add(pin.id)
            for key, rel, saved in extras:
                try:
                    if rel and not _path_used(manifest, rel):
                        _delete_rel(folder, rel)
                    if saved.get("kind") == "model" and not _model_still_held(manifest):
                        _remove_model(app)
                except (SuiteInstallError, StoreError, OSError):
                    failed = True
                    manifest.components[key] = saved
            statuses = _release_statuses(manifest, pins, removed, failed_detail)
    except SharedManifestError as exc:
        raise SuiteInstallError(str(exc)) from exc
    return RuntimeReport(not failed, False, tuple(statuses))


def cpu_onnxruntime_shadows_gpu() -> str | None:
    """Error when the CPU ``onnxruntime`` distribution is installed beside the GPU build.

    Both distributions provide the ``onnxruntime`` import. The CPU wheel then
    wins, and the providers are Azure and CPU only.
    """
    names: set[str] = set()
    for dist in importlib_metadata.distributions():
        metadata = dist.metadata
        raw = metadata.get("Name") if metadata else None
        if isinstance(raw, str) and raw.strip():
            names.add(raw.lower())
    if "onnxruntime" in names and "onnxruntime-gpu" in names:
        return (
            "CPU onnxruntime is installed beside onnxruntime-gpu, so the providers "
            "are Azure and CPU only. Uninstall it: python -m pip uninstall -y onnxruntime"
        )
    return None


def install_gpu_wheels(python: str, progress: Progress | None = None) -> None:
    """Install torch for CUDA 13, ONNX Runtime GPU 1.31, and TensorRT 10.14.

    ``GPU_EXTRA`` is the ``gpu`` extra. This install always applies those pins.
    The CPU ``onnxruntime`` wheel is forbidden: it shadows ``onnxruntime-gpu``
    and leaves only the CPU and Azure providers. Unpinned ``tensorrt-cu13`` is
    TensorRT 11 (``nvinfer_11``). ONNX Runtime 1.31 loads ``nvinfer_10`` from
    ``tensorrt-cu13-libs`` 10.14.1.48, recorded as component ``tensorrt-10``.
    Bindings for that TensorRT have no cp314 wheel, so the meta package is
    recorded with ``--no-deps``. Torch from the CUDA 13 index is ``torch-cu130``.
    """
    if getattr(sys, "frozen", False):
        raise SuiteInstallError("install the app directory, then run voxprint-enlarge --install-gpu")
    if not python:
        raise SuiteInstallError("a Python executable is required")
    _pip(python, ["install", "torch", "--index-url", TORCH_INDEX], "Installing torch for CUDA 13", progress, required=True)
    _pip(
        python,
        [
            "uninstall",
            "-y",
            "onnxruntime",
            "onnxruntime-gpu",
            "tensorrt",
            "tensorrt-cu13",
            "tensorrt-cu13-bindings",
            "tensorrt-cu13-libs",
        ],
        "Removing the CPU ONNX Runtime wheel and TensorRT 11",
        progress,
        required=False,
    )
    _pip(
        python,
        ["install", "--extra-index-url", NVIDIA_INDEX, *GPU_EXTRA, SPANDREL_SPEC, ONNX_SPEC],
        "Installing ONNX Runtime GPU 1.31 and TensorRT 10.14",
        progress,
        required=True,
    )
    _pip(
        python,
        ["uninstall", "-y", "onnxruntime"],
        "Removing the CPU ONNX Runtime wheel",
        progress,
        required=False,
    )
    _pip(
        python,
        ["install", "--no-deps", TRT_META_SPEC],
        "Recording the TensorRT 10.14 package without its bindings",
        progress,
        required=True,
    )


def bundled_python_tree() -> Path | None:
    """The ``python`` directory this process was started from, when it is a bundle."""
    executable = Path(sys.executable).resolve()
    parent = executable.parent
    if parent.name == "bin" and parent.parent.name == "python":
        return parent.parent
    if sys.platform == "win32" and parent.name == "python" and executable.name.lower() in {"python.exe", "pythonw.exe"}:
        return parent
    return None


def _skipped_report() -> RuntimeReport:
    items = tuple(ComponentStatus(pin.id, pin.version, "skipped", (), "CI stand-in") for pin in _pins())
    return RuntimeReport(True, True, items)


def _check_app(app: str) -> None:
    if _APP_ID.fullmatch(app) is None:
        raise SuiteInstallError("app id must be a short lowercase name")


def _emit(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _entry_refs(entry: dict[str, object] | None) -> list[str]:
    if not isinstance(entry, dict):
        return []
    value = entry.get("refs")
    if not isinstance(value, list):
        return []
    seen: list[str] = []
    for item in value:
        name = str(item).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _foreign_id(manifest: SharedManifest, component_id: str, app: str) -> bool:
    return any(name != app for name in _entry_refs(manifest.components.get(component_id)))


def _assign(manifest: SharedManifest, app: str, pins: tuple[_Pin, ...]) -> list[tuple[_Pin, str, str]]:
    python_pin = next(pin for pin in pins if pin.unit == "python")
    wheels = [pin for pin in pins if pin.unit == "gpu"]
    conflict = _python_conflict(manifest, app, python_pin, wheels)
    assigned: list[tuple[_Pin, str, str]] = []
    if conflict:
        rel = f"python-{python_pin.sha256[:12]}"
        for pin in pins:
            if pin.unit == "model":
                assigned.append(_model_assignment(manifest, app, pin))
            else:
                assigned.append((pin, f"{pin.id}-{python_pin.sha256[:12]}", rel))
        return assigned
    matched = find_reusable(manifest, python_pin.id, python_pin.version, python_pin.sha256)
    py_path = str(matched.get("path") or "python") if matched is not None else "python"
    for pin in pins:
        if pin.unit == "model":
            assigned.append(_model_assignment(manifest, app, pin))
        else:
            assigned.append((pin, pin.id, py_path))
    return assigned


def _python_conflict(manifest: SharedManifest, app: str, python_pin: _Pin, wheels: list[_Pin]) -> bool:
    matched = find_reusable(manifest, python_pin.id, python_pin.version, python_pin.sha256)
    if matched is None and _foreign_id(manifest, python_pin.id, app):
        return True
    for pin in wheels:
        if find_reusable(manifest, pin.id, pin.version, pin.sha256) is not None:
            continue
        if _foreign_id(manifest, pin.id, app):
            return True
    return False


def _model_assignment(manifest: SharedManifest, app: str, pin: _Pin) -> tuple[_Pin, str, str]:
    found = find_reusable(manifest, pin.id, pin.version, pin.sha256)
    if found is not None:
        return pin, pin.id, str(found.get("path") or pin.path)
    if _foreign_id(manifest, pin.id, app):
        suffix = pin.sha256[:12] or "unset"
        return pin, f"{pin.id}-{suffix}", f"{pin.path}-{suffix}"
    return pin, pin.id, pin.path


def _unit_path(assignments: list[tuple[_Pin, str, str]], unit: str) -> str:
    for pin, _component_id, path in assignments:
        if pin.unit == unit:
            return path
    return "python"


def _unit_ready(
    manifest: SharedManifest,
    assignments: list[tuple[_Pin, str, str]],
    unit: ComponentId,
    folder: Path,
    probes: Mapping[str, Probe] | None,
) -> bool:
    rel = _unit_path(assignments, unit)
    if not _call_probe(unit, folder, probes, rel):
        return False
    for pin, component_id, _path in assignments:
        if pin.unit != unit:
            continue
        if find_reusable(manifest, component_id, pin.version, pin.sha256) is None:
            return False
    return True


def _path_used(manifest: SharedManifest, rel: str) -> bool:
    if not rel:
        return True
    for entry in manifest.components.values():
        if isinstance(entry, dict) and str(entry.get("path") or "") == rel:
            return True
    return False


def _model_still_held(manifest: SharedManifest) -> bool:
    for entry in manifest.components.values():
        if isinstance(entry, dict) and entry.get("kind") == "model" and _entry_refs(entry):
            return True
    return False


def _unregister_quietly(app: str) -> None:
    try:
        unregister(app)
    except (OSError, StoreError):
        return


def _delete_rel(root: Path, rel: str) -> None:
    if not rel or ".." in Path(rel).parts:
        raise SuiteInstallError("refusing to delete a path outside the shared folder")
    dest = (root / rel).resolve()
    base = root.resolve()
    if dest != base and base not in dest.parents:
        raise SuiteInstallError("refusing to delete a path outside the shared folder")
    if dest == base:
        raise SuiteInstallError("refusing to delete the shared folder itself")
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
        return
    if dest.is_dir():
        shutil.rmtree(dest)


def _release_statuses(
    manifest: SharedManifest,
    pins: tuple[_Pin, ...],
    removed: set[str],
    failed_detail: dict[str, str],
) -> list[ComponentStatus]:
    statuses: list[ComponentStatus] = []
    for pin in pins:
        if pin.unit in failed_detail and pin.id in manifest.components:
            holders = tuple(_entry_refs(manifest.components.get(pin.id)))
            statuses.append(ComponentStatus(pin.id, pin.version, "failed", holders, failed_detail[pin.unit]))
            continue
        if pin.id in removed:
            statuses.append(ComponentStatus(pin.id, pin.version, "ok", (), "removed"))
            continue
        entry = manifest.components.get(pin.id)
        holders = tuple(_entry_refs(entry))
        if holders:
            statuses.append(ComponentStatus(pin.id, pin.version, "ok", holders, "held"))
        else:
            statuses.append(ComponentStatus(pin.id, pin.version, "ok", (), "absent"))
    return statuses


def _call_probe(
    component: ComponentId,
    root: Path,
    probes: Mapping[str, Probe] | None,
    rel: str = "python",
) -> bool:
    if probes is not None and component in probes:
        return probes[component](root)
    if component == "python":
        return _python_ready(root, rel)
    if component == "gpu":
        return _gpu_ready(root, rel)
    if component == "model":
        return _model_ready()
    assert_never(component)


def _call_installer(
    component: ComponentId,
    root: Path,
    progress: Progress | None,
    installers: Mapping[str, Installer] | None,
    rel: str = "python",
) -> None:
    if installers is not None and component in installers:
        installers[component](root, progress or (lambda _message: None))
        return
    sink = progress or (lambda _message: None)
    if component == "python":
        _install_python(root, rel, sink)
        return
    if component == "gpu":
        _install_gpu(root, rel, sink)
        return
    if component == "model":
        _install_model(sink)
        return
    assert_never(component)


def _call_remover(
    component: ComponentId,
    root: Path,
    app: str,
    removers: Mapping[str, Remover] | None,
    rel: str = "python",
) -> None:
    if removers is not None:
        if component not in removers:
            raise SuiteInstallError(f"no remover for {component}")
        removers[component](root)
        return
    if component == "python":
        _remove_python(root, rel)
        return
    if component == "gpu":
        _remove_gpu(root, rel)
        return
    if component == "model":
        _remove_model(app)
        return
    assert_never(component)


def _register_model_user(app: str) -> None:
    try:
        register(app)
    except (OSError, StoreError):
        return


def _python_executable(root: Path, rel: str = "python") -> str | None:
    folder = root / rel
    if sys.platform == "win32":
        candidate = folder / "python.exe"
    else:
        candidate = folder / "bin" / "python3"
    if candidate.is_file():
        return str(candidate)
    pointer = folder / "interpreter.json"
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    executable = data.get("executable")
    version = str(data.get("version") or "")
    if not isinstance(executable, str) or not Path(executable).is_file():
        return None
    if version != PYTHON_VERSION and not version.startswith(f"{PYTHON_VERSION}."):
        return None
    return executable


def _python_ready(root: Path, rel: str = "python") -> bool:
    executable = _python_executable(root, rel)
    if executable is None:
        return False
    script = "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)"
    completed = subprocess.run([executable, "-c", script], check=False, capture_output=True)
    return completed.returncode == 0


def _install_python(root: Path, rel: str, progress: Progress) -> None:
    dest = root / rel
    bundled = bundled_python_tree()
    if bundled is not None:
        try:
            same = bundled.resolve() == dest.resolve()
        except OSError:
            same = False
        if not same:
            progress("Copying Python 3.14 into the shared suite folder")
            if dest.is_symlink():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
            try:
                shutil.copytree(bundled, dest, symlinks=True)
            except OSError as exc:
                raise SuiteInstallError(f"could not copy Python 3.14: {exc}") from exc
            if _python_ready(root, rel):
                return
            if dest.is_symlink():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
    if sys.version_info[:2] == (3, 14):
        progress("Recording the current Python 3.14 interpreter")
        dest.mkdir(parents=True, exist_ok=True)
        payload = {"executable": sys.executable, "version": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"}
        (dest / "interpreter.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return
    raise SuiteInstallError("Python 3.14 was not found. Install the Enlarge bundle, which carries that runtime.")


def _remove_python(root: Path, rel: str = "python") -> None:
    dest = root / rel
    if dest.is_symlink():
        dest.unlink()
        return
    if not dest.exists():
        return
    if root.resolve() not in dest.resolve().parents:
        raise SuiteInstallError("refusing to delete a Python outside the shared folder")
    shutil.rmtree(dest)


def _gpu_ready(root: Path, rel: str = "python") -> bool:
    python = _python_executable(root, rel) or sys.executable
    if not Path(python).is_file():
        return False
    completed = subprocess.run([python, "-c", _GPU_PROBE], check=False, capture_output=True)
    return completed.returncode == 0


def _install_gpu(root: Path, rel: str, progress: Progress) -> None:
    python = _python_executable(root, rel) or sys.executable
    install_gpu_wheels(python, progress)


def _remove_gpu(root: Path, rel: str = "python") -> None:
    python = _python_executable(root, rel) or sys.executable
    if python and Path(python).is_file() and not getattr(sys, "frozen", False):
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "torch",
                "onnxruntime",
                "onnxruntime-gpu",
                "tensorrt-cu13",
                "tensorrt",
                "tensorrt-cu13-bindings",
                "tensorrt-cu13-libs",
                "spandrel",
                "onnx",
            ],
            check=False,
        )


def _default_model() -> ModelManifest:
    catalog = load_builtin(release=True)
    model = catalog.get(DEFAULT_MODEL_ID)
    if model is None:
        raise SuiteInstallError("the default model is not in the catalogue")
    return model


def _model_ready() -> bool:
    model = _default_model()
    if cached_onnx(model) is None:
        return False
    if model.onnx_url and model.onnx_sha256:
        return True
    state, _detail = weight_report(model)
    return state == "ready"


def _install_model(progress: Progress) -> None:
    model = _default_model()

    def on_progress(_fraction: float, message: str) -> None:
        progress(message)

    try:
        ensure_onnx(model, on_progress=on_progress)
    except (StoreError, OnnxExportError) as exc:
        raise SuiteInstallError(str(exc)) from exc


def _remove_model(app: str) -> None:
    others = unregister(app)
    if others:
        return
    model = _default_model()
    if not model.filename or "/" in model.filename or "\\" in model.filename:
        raise SuiteInstallError(f"{model.id} has an unsafe filename")
    target = models_dir() / model.filename
    target.unlink(missing_ok=True)
    target.with_name(target.name + ".partial").unlink(missing_ok=True)
    cached = cached_onnx(model)
    if cached is not None:
        cached.unlink(missing_ok=True)
    if model.sha256:
        (models_dir() / f"{model.sha256.lower()}.onnx").unlink(missing_ok=True)


def _pip(python: str, args: list[str], message: str, progress: Progress | None, *, required: bool) -> None:
    _emit(progress, message)
    completed = subprocess.run([python, "-m", "pip", *args], check=False)
    if required and completed.returncode != 0:
        raise SuiteInstallError(f"{message} failed ({completed.returncode})")
