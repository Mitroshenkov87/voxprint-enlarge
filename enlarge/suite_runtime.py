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

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, assert_never

from voxprint_common.atomic import LockTimeout, exclusive_lock, write_json
from voxprint_common.paths import home, models_dir, shared_dir
from voxprint_common.runtime_manifest import (
    ManifestError,
    load as load_manifest,
    lock_path as manifest_lock_path,
)
from enlarge.paths import voxprint_home_source, voxprint_home_warning
from voxprint_engine.onnx_export import OnnxExportError, cached_onnx, ensure_onnx
from voxprint_engine.providers import ci_lanczos
from voxprint_engine.registry import DEFAULT_MODEL_ID, ModelManifest, load_builtin
from voxprint_engine.store import StoreError, register, unregister, weight_report

# Pin text whose sha256 is the lock hash Enlarge writes for these components.
# voxprint-common 0.1 records a component digest; it does not ship these pins.
PYTHON_LOCK = "python==3.14"
TORCH_LOCK = "torch@cu130"
ORT_LOCK = "onnxruntime-gpu==1.31.0"
TRT_LOCK = "tensorrt-cu13-libs==10.14.1.48"
_KIND_OK = frozenset({"runtime", "wheels", "model", "tool"})
_COMPONENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_APP_REF = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_RECORD_FIELDS = frozenset({"kind", "version", "path", "sha256", "installed_at", "refs"})


class _ManifestTxError(RuntimeError):
    """The shared manifest could not be read, locked, or written."""


@dataclass
class _ManifestView:
    """Mutable schema 1 document used for one installer transaction."""

    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON object. Unknown top-level keys from ``extra`` are included."""
        body = dict(self.extra)
        body["schema"] = 1
        body["components"] = self.components
        return body


def _lock_hash(text: str) -> str:
    """SHA-256 hex of a pin string, used as a component's lock hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_path(root: Path | None) -> Path:
    folder = root or shared_dir()
    return folder / "manifest.json"


def _view_from_common(path: Path) -> _ManifestView:
    """Read a schema 1 file that :func:`voxprint_common.runtime_manifest.load` already migrated."""
    if not path.is_file():
        return _ManifestView()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise _ManifestTxError(f"{path.name} could not be read") from exc
    if not isinstance(payload, dict):
        raise _ManifestTxError(f"{path.name} must be a JSON object")
    raw = payload.get("components")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise _ManifestTxError("manifest components must be an object")
    components = {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}
    extra = {key: value for key, value in payload.items() if key not in {"schema", "components"}}
    return _ManifestView(components, extra)


def _view_load(root: Path | None = None, *, timeout: float = 30.0) -> _ManifestView:
    """Load ``manifest.json``, migrating an older shape through ``voxprint_common``."""
    path = _manifest_path(root)
    if not path.is_file():
        return _ManifestView()
    try:
        load_manifest(path, timeout=timeout)
    except ManifestError as exc:
        raise _ManifestTxError(str(exc)) from exc
    except LockTimeout as exc:
        raise _ManifestTxError(str(exc)) from exc
    return _view_from_common(path)


@contextmanager
def _view_update(root: Path | None = None, *, timeout: float = 30.0) -> Iterator[_ManifestView]:
    """Hold ``manifest.lock``, yield the manifest, and save it when the body returns.

    Migration runs first, through :func:`voxprint_common.runtime_manifest.load`.
    The lock is not re-entrant, so the write takes it only after that call returns.
    """
    folder = root or shared_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    if path.is_file():
        try:
            load_manifest(path, timeout=timeout)
        except ManifestError as exc:
            raise _ManifestTxError(str(exc)) from exc
        except LockTimeout as exc:
            raise _ManifestTxError(str(exc)) from exc
    try:
        with exclusive_lock(manifest_lock_path(path), timeout=timeout):
            view = _view_from_common(path)
            try:
                yield view
            except BaseException:
                raise
            else:
                write_json(path, view.as_dict())
    except LockTimeout as exc:
        raise _ManifestTxError(str(exc)) from exc


def _find_reusable(manifest: _ManifestView, component_id: str, version: str, sha256: str) -> dict[str, Any] | None:
    """The component when id, version, and sha256 all match, else None."""
    entry = manifest.components.get(component_id)
    if not isinstance(entry, dict):
        return None
    if str(entry.get("version") or "") != version:
        return None
    if str(entry.get("sha256") or "") != sha256:
        return None
    return entry


def _ref_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: list[str] = []
    for item in value:
        name = item.strip() if isinstance(item, str) else str(item).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _check_component_id(component_id: str) -> None:
    if _COMPONENT_ID.fullmatch(component_id) is None:
        raise _ManifestTxError(f"component id {component_id!r} is not allowed")


def _check_ref_app(app: str) -> None:
    if _APP_REF.fullmatch(app) is None:
        raise _ManifestTxError("app id must be a short lowercase name")


def _add_ref(manifest: _ManifestView, component_id: str, app: str) -> dict[str, Any]:
    _check_component_id(component_id)
    _check_ref_app(app)
    entry = manifest.components.get(component_id)
    if not isinstance(entry, dict):
        raise _ManifestTxError(f"{component_id} is not in the manifest")
    refs = _ref_names(entry.get("refs"))
    if app not in refs:
        refs.append(app)
    entry["refs"] = refs
    return entry


def _drop_ref(manifest: _ManifestView, component_id: str, app: str) -> bool:
    """Drop ``app``. Return True when the component's ref list is now empty."""
    _check_component_id(component_id)
    _check_ref_app(app)
    entry = manifest.components.get(component_id)
    if not isinstance(entry, dict):
        return False
    refs = [name for name in _ref_names(entry.get("refs")) if name != app]
    entry["refs"] = refs
    return not refs


def _stamp(
    kind: str,
    version: str,
    path: str,
    sha256: str,
    installed_at: str | None,
    refs: list[str],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    unknown: dict[str, Any] = {}
    if previous:
        unknown = {key: value for key, value in previous.items() if key not in _RECORD_FIELDS}
    record: dict[str, Any] = dict(unknown)
    record.update(
        {
            "installed_at": installed_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind,
            "path": path,
            "refs": list(refs),
            "sha256": sha256,
            "version": version,
        }
    )
    return record


def _side_path(path: str, sha256: str) -> str:
    suffix = (sha256 or "unset")[:12]
    base = path or "runtime"
    return f"{base}-{suffix}"


def _foreign(entry: dict[str, Any], app: str) -> bool:
    return any(name != app for name in _ref_names(entry.get("refs")))


def _reserve(
    manifest: _ManifestView,
    *,
    component_id: str,
    kind: str,
    version: str,
    path: str,
    sha256: str,
    app: str,
    installed_at: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Reuse a matching component, or record ``app`` on a side-by-side path.

    ``voxprint_common.runtime_manifest.add_ref`` refuses a different digest
    while another app still refs the id. The installer keeps the occupied
    record and writes the new copy under ``{id}-{sha256[:12]}``.
    """
    _check_component_id(component_id)
    if kind not in _KIND_OK:
        raise _ManifestTxError(f"component kind {kind!r} is not allowed")
    _check_ref_app(app)
    if not _path_allowed(path, kind):
        raise _ManifestTxError("component path must stay inside the shared directory or the shared models directory")
    found = _find_reusable(manifest, component_id, version, sha256)
    if found is not None:
        _add_ref(manifest, component_id, app)
        return found, True
    current = manifest.components.get(component_id)
    if isinstance(current, dict) and _foreign(current, app):
        suffix = (sha256 or "unset")[:12]
        side_id = f"{component_id}-{suffix}"
        _check_component_id(side_id)
        side = _find_reusable(manifest, side_id, version, sha256)
        if side is not None:
            _add_ref(manifest, side_id, app)
            return side, True
        existing = manifest.components.get(side_id)
        if isinstance(existing, dict) and _foreign(existing, app):
            raise _ManifestTxError(f"{component_id} is in use under another app")
        occupied = str(current.get("path") or "")
        side_path = path if path and path != occupied else _side_path(occupied or path, sha256)
        if not _path_allowed(side_path, kind):
            raise _ManifestTxError("component path must stay inside the shared directory or the shared models directory")
        if app in _ref_names(current.get("refs")):
            _drop_ref(manifest, component_id, app)
        record = _stamp(kind, version, side_path, sha256, installed_at, [app], existing if isinstance(existing, dict) else None)
        manifest.components[side_id] = record
        return record, False
    previous = current if isinstance(current, dict) else None
    refs = _ref_names(previous.get("refs")) if previous else []
    if app not in refs:
        refs.append(app)
    record = _stamp(kind, version, path, sha256, installed_at, refs, previous)
    manifest.components[component_id] = record
    return record, False


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
ONNX_SPEC = "onnx>=1.17,<2"

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


def _model_roots() -> tuple[Path, ...]:
    """Directories a model component path may use, matching ``voxprint_common`` 0.1.1."""
    roots = [models_dir(), shared_dir()]
    cache = os.environ.get("VOXPRINT_MODEL_CACHE", "").strip()
    if cache:
        folder = Path(os.path.expandvars(os.path.expanduser(cache)))
        if folder.is_absolute():
            roots.append(folder)
    return tuple(roots)


def _inside_model_root(candidate: Path) -> bool:
    cand = os.path.normcase(str(candidate.absolute()))
    for root in _model_roots():
        base = os.path.normcase(str(root.absolute()))
        if cand == base or cand.startswith(base + os.sep):
            return True
    return False


def _path_allowed(path: str, kind: str) -> bool:
    """Relative paths stay in the shared tree. A model may be absolute under the models root."""
    if not path or ".." in Path(path).parts:
        return False
    candidate = Path(path)
    if candidate.is_absolute() or path.startswith(("/", "\\")):
        return kind == "model" and _inside_model_root(candidate)
    return True


def _model_payload_path() -> str:
    """Absolute path of the default weight under the shared models directory."""
    model = _default_model()
    filename = model.filename.strip()
    if filename and "/" not in filename and "\\" not in filename:
        return str((models_dir() / filename).absolute())
    return str(models_dir().absolute())


def _model_pin() -> _Pin:
    digest = _default_model().sha256.lower()
    prefix = digest[:12]
    return _Pin(f"model-{prefix}", "model", DEFAULT_MODEL_ID, _model_payload_path(), digest, "model")


def default_model_component_id() -> str:
    """Manifest id Enlarge uses for its default upscale model."""
    return _model_pin().id


def _pins() -> tuple[_Pin, ...]:
    return (
        _Pin("python-3.14", "runtime", PYTHON_VERSION, "python", _lock_hash(PYTHON_LOCK), "python"),
        _Pin("torch-cu130", "wheels", GPU_VERSION, "python", _lock_hash(TORCH_LOCK), "gpu"),
        _Pin("onnxruntime-gpu-1.31", "wheels", "1.31.0", "python", _lock_hash(ORT_LOCK), "gpu"),
        _Pin("tensorrt-10", "wheels", "10.14.1.48", "python", _lock_hash(TRT_LOCK), "gpu"),
    )


def managed_ids() -> tuple[str, ...]:
    """Component ids ``ensure_runtime`` records in the shared manifest."""
    return tuple(pin.id for pin in _pins())


def read_manifest(root: Path | None = None) -> dict[str, object]:
    """Return the suite manifest. A missing or damaged file is an empty schema 1 document.

    An old Canvas or Enlarge file is migrated to schema 1 before it is returned.
    """
    try:
        return _view_load(root).as_dict()
    except _ManifestTxError as exc:
        raise SuiteInstallError(str(exc)) from exc


def ensure_runtime(
    app_id: str,
    *,
    repair: bool = True,
    progress: Progress | None = None,
    root: Path | None = None,
    probes: Mapping[str, Probe] | None = None,
    installers: Mapping[str, Installer] | None = None,
    allow_skip: bool = True,
) -> RuntimeReport:
    """Hold ``app_id`` on the shared runtime and wheels, installing anything missing.

    ``app_id`` is written into ``refs`` on every runtime and wheel component
    this call installs or reuses. The same id a second time does not add a
    second ref. This function does not download Enlarge's default model.
    ``probes`` and ``installers`` replace the real checks for tests. The real
    steps copy or record Python 3.14 and install the CUDA 13 wheels (torch
    cu130, ONNX Runtime GPU, TensorRT 10). A component another app refs is
    left in place and the new copy is recorded beside it.
    """
    app = app_id
    _check_app(app)
    _emit(progress, f"Voxprint home: {home()} ({voxprint_home_source()})")
    warning = voxprint_home_warning()
    if warning:
        _emit(progress, warning)
    if allow_skip and ci_lanczos():
        return _skipped_report()
    folder = root or shared_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pins = _pins()
    try:
        snapshot = _view_load(folder)
    except _ManifestTxError as exc:
        raise SuiteInstallError(str(exc)) from exc
    assignments = _assign(snapshot, app, pins)
    outcomes: dict[str, tuple[str, str]] = {}
    units: tuple[ComponentId, ...] = tuple(dict.fromkeys(pin.unit for pin in pins))
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
        except (SuiteInstallError, StoreError, OSError, _ManifestTxError) as exc:
            outcomes[unit] = ("failed", str(exc))
            continue
        if _call_probe(unit, folder, probes, rel):
            outcomes[unit] = ("repaired", "repaired")
        else:
            outcomes[unit] = ("failed", "still missing after install")
    records: dict[str, dict[str, object]] = {}
    if any(state in {"ok", "repaired"} for state, _detail in outcomes.values()):
        try:
            with _view_update(folder) as manifest:
                for pin, component_id, path in assignments:
                    state, _detail = outcomes[pin.unit]
                    if state not in {"ok", "repaired"}:
                        continue
                    record, _reused = _reserve(
                        manifest,
                        component_id=component_id,
                        kind=pin.kind,
                        version=pin.version,
                        path=path,
                        sha256=pin.sha256,
                        app=app,
                    )
                    records[component_id] = record
        except _ManifestTxError as exc:
            raise SuiteInstallError(str(exc)) from exc
    statuses: list[ComponentStatus] = []
    for pin, component_id, _path in assignments:
        state, detail = outcomes[pin.unit]
        holders = tuple(_entry_refs(records.get(component_id)))
        if state == "ok":
            _emit(progress, f"{component_id} {pin.version} is ready")
        statuses.append(ComponentStatus(component_id, pin.version, state, holders, detail))
    ok = all(item.state in {"ok", "repaired"} for item in statuses)
    return RuntimeReport(ok, False, tuple(statuses))


def ensure_default_model(
    app_id: str,
    *,
    repair: bool = True,
    progress: Progress | None = None,
    root: Path | None = None,
    probes: Mapping[str, Probe] | None = None,
    installers: Mapping[str, Installer] | None = None,
    allow_skip: bool = True,
) -> RuntimeReport:
    """Download Enlarge's default model and record ``app_id`` in its refs.

    Any ``app_id`` other than ``enlarge`` returns without installing. The
    shared runtime call does not download this model.
    """
    _check_app(app_id)
    pin = _model_pin()
    if app_id != "enlarge":
        return RuntimeReport(True, True, (ComponentStatus(pin.id, pin.version, "skipped", (), "not this app"),))
    if allow_skip and ci_lanczos():
        return RuntimeReport(True, True, (ComponentStatus(pin.id, pin.version, "skipped", (), "CI stand-in"),))
    folder = root or shared_dir()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _view_load(folder)
    except _ManifestTxError as exc:
        raise SuiteInstallError(str(exc)) from exc
    _pin, component_id, rel = _model_assignment(snapshot, app_id, pin)
    assignments = [(pin, component_id, rel)]
    _emit(progress, f"Checking {component_id} {pin.version}")
    if _unit_ready(snapshot, assignments, "model", folder, probes):
        state, detail = "ok", "ready"
    elif not repair:
        state, detail = "missing", "not installed"
    else:
        _emit(progress, f"Installing {component_id} {pin.version}")
        try:
            _call_installer("model", folder, progress, installers, rel)
        except (SuiteInstallError, StoreError, OSError, _ManifestTxError) as exc:
            return RuntimeReport(False, False, (ComponentStatus(component_id, pin.version, "failed", (), str(exc)),))
        if _call_probe("model", folder, probes, rel):
            state, detail = "repaired", "repaired"
        else:
            state, detail = "failed", "still missing after install"
    holders: tuple[str, ...] = ()
    if state in {"ok", "repaired"}:
        try:
            with _view_update(folder) as manifest:
                record, _reused = _reserve(
                    manifest,
                    component_id=component_id,
                    kind=pin.kind,
                    version=pin.version,
                    path=rel,
                    sha256=pin.sha256,
                    app=app_id,
                )
                holders = tuple(_entry_refs(record))
        except _ManifestTxError as exc:
            raise SuiteInstallError(str(exc)) from exc
        _register_model_user(app_id)
        if state == "ok":
            _emit(progress, f"{component_id} {pin.version} is ready")
    ready = state in {"ok", "repaired"}
    return RuntimeReport(ready, False, (ComponentStatus(component_id, pin.version, state, holders, detail),))


def release_runtime(
    app_id: str,
    *,
    progress: Progress | None = None,
    root: Path | None = None,
    removers: Mapping[str, Remover] | None = None,
) -> RuntimeReport:
    """Drop ``app_id`` from the manifest. Delete a payload only when its ref list is empty."""
    app = app_id
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
        with _view_update(folder) as manifest:
            pending: list[tuple[_Pin, str, dict[str, object]]] = []
            for pin in pins:
                entry = manifest.components.get(pin.id)
                if not isinstance(entry, dict) or app not in _entry_refs(entry):
                    continue
                others = [name for name in _entry_refs(entry) if name != app]
                if others:
                    _drop_ref(manifest, pin.id, app)
                    if pin.unit == "model":
                        _unregister_quietly(app)
                    _emit(progress, f"{pin.id} {pin.version} still used by {', '.join(others)}")
                    continue
                saved = dict(entry)
                saved["refs"] = list(_entry_refs(entry))
                rel = str(entry.get("path") or pin.path)
                _drop_ref(manifest, pin.id, app)
                del manifest.components[pin.id]
                pending.append((pin, rel, saved))
            extras: list[tuple[str, str, dict[str, object]]] = []
            for key, entry in list(manifest.components.items()):
                if key in known or not isinstance(entry, dict) or app not in _entry_refs(entry):
                    continue
                others = [name for name in _entry_refs(entry) if name != app]
                if others:
                    _drop_ref(manifest, key, app)
                    if entry.get("kind") == "model":
                        _unregister_quietly(app)
                    continue
                saved = dict(entry)
                saved["refs"] = list(_entry_refs(entry))
                rel = str(entry.get("path") or "")
                _drop_ref(manifest, key, app)
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
    except _ManifestTxError as exc:
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


def _foreign_id(manifest: _ManifestView, component_id: str, app: str) -> bool:
    return any(name != app for name in _entry_refs(manifest.components.get(component_id)))


def _assign(manifest: _ManifestView, app: str, pins: tuple[_Pin, ...]) -> list[tuple[_Pin, str, str]]:
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
    matched = _find_reusable(manifest, python_pin.id, python_pin.version, python_pin.sha256)
    py_path = str(matched.get("path") or "python") if matched is not None else "python"
    for pin in pins:
        if pin.unit == "model":
            assigned.append(_model_assignment(manifest, app, pin))
        else:
            assigned.append((pin, pin.id, py_path))
    return assigned


def _python_conflict(manifest: _ManifestView, app: str, python_pin: _Pin, wheels: list[_Pin]) -> bool:
    matched = _find_reusable(manifest, python_pin.id, python_pin.version, python_pin.sha256)
    if matched is None and _foreign_id(manifest, python_pin.id, app):
        return True
    for pin in wheels:
        if _find_reusable(manifest, pin.id, pin.version, pin.sha256) is not None:
            continue
        if _foreign_id(manifest, pin.id, app):
            return True
    return False


def _model_assignment(manifest: _ManifestView, app: str, pin: _Pin) -> tuple[_Pin, str, str]:
    found = _find_reusable(manifest, pin.id, pin.version, pin.sha256)
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
    manifest: _ManifestView,
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
        if _find_reusable(manifest, component_id, pin.version, pin.sha256) is None:
            return False
    return True


def _path_used(manifest: _ManifestView, rel: str) -> bool:
    if not rel:
        return True
    for entry in manifest.components.values():
        if isinstance(entry, dict) and str(entry.get("path") or "") == rel:
            return True
    return False


def _model_still_held(manifest: _ManifestView) -> bool:
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
    dest = _shared_child(root, rel)
    base = root.resolve()
    if dest == base:
        raise SuiteInstallError("refusing to delete the shared folder itself")
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
        return
    if dest.is_dir():
        shutil.rmtree(dest)


def _release_statuses(
    manifest: _ManifestView,
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


def _shared_child(root: Path, rel: str) -> Path:
    """Resolve ``rel`` inside ``root``. Absolute paths and ``..`` are refused."""
    if not rel or Path(rel).is_absolute() or rel.startswith(("/", "\\")) or ".." in Path(rel).parts:
        raise SuiteInstallError("refusing to use a path outside the shared folder")
    base = root.resolve()
    dest = (base / rel).resolve()
    if dest != base and base not in dest.parents:
        raise SuiteInstallError("refusing to use a path outside the shared folder")
    return dest


def _python_executable(root: Path, rel: str = "python") -> str | None:
    try:
        folder = _shared_child(root, rel)
    except SuiteInstallError:
        return None
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
    dest = _shared_child(root, rel)
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
    dest = _shared_child(root, rel)
    if not dest.exists() and not dest.is_symlink():
        return
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
        return
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
