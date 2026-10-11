"""Schema 1 manifest for the shared Voxprint runtime.

The file is ``%LOCALAPPDATA%\\Voxprint\\shared\\manifest.json`` on Windows and
``~/.local/share/voxprint/shared/manifest.json`` elsewhere. Canvas, Clarity,
and Enlarge all take a ref on the components they use. A component is removed
only when ``refs`` is empty.

``load`` reads that file and, when it still has the old Canvas ``version`` /
``resources`` shape or the old Enlarge ``holders`` shape, writes it back as
schema 1. Unknown keys are kept. ``update`` is the read-modify-write path:
it holds ``manifest.lock`` (30 seconds) and saves with ``manifest.json.tmp``,
an fsync, and a rename.

A component is reused only when its id, version, and sha256 all match.
Otherwise the caller installs beside it. ``reserve`` never overwrites a
component another app still refs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voxprint_engine.paths import shared_dir
from voxprint_engine.registry import DEFAULT_MODEL_ID, load_builtin

SCHEMA = 1
LOCK_NAME = "manifest.lock"
LOCK_TIMEOUT_S = 30.0
KINDS = frozenset({"runtime", "wheels", "model", "tool"})
_FIELDS = frozenset({"kind", "version", "path", "sha256", "installed_at", "refs"})
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_APP = re.compile(r"^[a-z][a-z0-9-]{0,31}$")

# Pin text whose sha256 is the lock hash Enlarge writes for these components.
PYTHON_LOCK = "python==3.14"
TORCH_LOCK = "torch@cu130"
ORT_LOCK = "onnxruntime-gpu==1.31.0"
TRT_LOCK = "tensorrt-cu13-libs==10.14.1.48"


class SharedManifestError(RuntimeError):
    """The shared manifest could not be read, locked, or written."""

    def __init__(self, message: str, *, kind: str = "error") -> None:
        """Store ``kind`` so a caller can tell a schema error from a lock timeout."""
        super().__init__(message)
        self.kind = kind


@dataclass
class SharedManifest:
    """One schema 1 document. Component records are the dicts ``components`` holds."""

    schema: int = SCHEMA
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON object. Unknown top-level keys from ``extra`` are included."""
        body = dict(self.extra)
        body["schema"] = SCHEMA
        body["components"] = self.components
        return body


def lock_hash(text: str) -> str:
    """SHA-256 hex of a pin string, used as a component's lock hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(root: Path | None = None, *, timeout: float = LOCK_TIMEOUT_S) -> SharedManifest:
    """Read the manifest. Migrate an old Canvas or Enlarge file and write it back.

    A missing or damaged file is an empty schema 1 document and is not created.
    ``timeout`` is how long to wait for ``manifest.lock`` when a migration has
    to save.
    """
    folder = root or shared_dir()
    path = folder / "manifest.json"
    if not path.is_file():
        return SharedManifest()
    data = _parse(path)
    if data is None:
        return SharedManifest()
    if not _needs_migration(data):
        return _from_new(data)
    with _locked(folder, timeout):
        fresh = _parse(path)
        if fresh is None:
            return SharedManifest()
        if not _needs_migration(fresh):
            return _from_new(fresh)
        manifest = _migrate(fresh)
        _write(folder, manifest)
        return manifest


def save(manifest: SharedManifest, root: Path | None = None, *, timeout: float = LOCK_TIMEOUT_S) -> None:
    """Write ``manifest`` atomically while holding ``manifest.lock``.

    The write is ``manifest.json.tmp``, then fsync, then rename over
    ``manifest.json``. Prefer :func:`update` for a read-modify-write.
    """
    folder = root or shared_dir()
    folder.mkdir(parents=True, exist_ok=True)
    with _locked(folder, timeout):
        _write(folder, manifest)


@contextmanager
def update(root: Path | None = None, *, timeout: float = LOCK_TIMEOUT_S) -> Iterator[SharedManifest]:
    """Hold ``manifest.lock``, yield the manifest, and save it if the body returns.

    An exception from the body leaves the file as it was. ``timeout`` is the
    wait for the lock, 30 seconds unless a test passes a shorter one.
    """
    folder = root or shared_dir()
    folder.mkdir(parents=True, exist_ok=True)
    with _locked(folder, timeout):
        manifest = _read_unlocked(folder)
        try:
            yield manifest
        except BaseException:
            raise
        else:
            _write(folder, manifest)


def find_reusable(
    manifest: SharedManifest,
    component_id: str,
    version: str,
    sha256: str,
) -> dict[str, Any] | None:
    """The component when id, version, and sha256 all match, else None."""
    entry = manifest.components.get(component_id)
    if not isinstance(entry, dict):
        return None
    if str(entry.get("version") or "") != version:
        return None
    if str(entry.get("sha256") or "") != sha256:
        return None
    return entry


def add_ref(manifest: SharedManifest, component_id: str, app: str) -> dict[str, Any]:
    """Append ``app`` to ``component_id``. A second add does not duplicate it."""
    _check_id(component_id)
    _check_app(app)
    entry = manifest.components.get(component_id)
    if not isinstance(entry, dict):
        raise SharedManifestError(f"{component_id} is not in the manifest")
    refs = _ref_list(entry.get("refs"))
    if app not in refs:
        refs.append(app)
    entry["refs"] = refs
    return entry


def remove_ref(manifest: SharedManifest, component_id: str, app: str) -> bool:
    """Drop ``app``. Return True when the component's ref list is now empty."""
    _check_id(component_id)
    _check_app(app)
    entry = manifest.components.get(component_id)
    if not isinstance(entry, dict):
        return False
    refs = [name for name in _ref_list(entry.get("refs")) if name != app]
    entry["refs"] = refs
    return not refs


def reserve(
    manifest: SharedManifest,
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

    The bool is True when id, version, and sha256 already matched. A component
    another app refs is not overwritten: the new record's id is
    ``{component_id}-{sha256[:12]}``. ``path`` is kept when it already differs
    from the occupied component's path.
    """
    _check_id(component_id)
    _check_kind(kind)
    _check_app(app)
    _check_rel(path)
    found = find_reusable(manifest, component_id, version, sha256)
    if found is not None:
        add_ref(manifest, component_id, app)
        return found, True
    current = manifest.components.get(component_id)
    if isinstance(current, dict) and _foreign(current, app):
        suffix = (sha256 or "unset")[:12]
        side_id = f"{component_id}-{suffix}"
        _check_id(side_id)
        side = find_reusable(manifest, side_id, version, sha256)
        if side is not None:
            add_ref(manifest, side_id, app)
            return side, True
        existing = manifest.components.get(side_id)
        if isinstance(existing, dict) and _foreign(existing, app):
            raise SharedManifestError(f"{component_id} is in use under another app")
        occupied = str(current.get("path") or "")
        side_path = path if path and path != occupied else _side_path(occupied or path, sha256)
        _check_rel(side_path)
        if app in _ref_list(current.get("refs")):
            remove_ref(manifest, component_id, app)
        record = _stamp(kind, version, side_path, sha256, installed_at, [app], existing if isinstance(existing, dict) else None)
        manifest.components[side_id] = record
        return record, False
    previous = current if isinstance(current, dict) else None
    refs = _ref_list(previous.get("refs")) if previous else []
    if app not in refs:
        refs.append(app)
    record = _stamp(kind, version, path, sha256, installed_at, refs, previous)
    manifest.components[component_id] = record
    return record, False


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
        unknown = {key: value for key, value in previous.items() if key not in _FIELDS}
    record: dict[str, Any] = dict(unknown)
    record.update(
        {
            "installed_at": installed_at or _now(),
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
    return any(name != app for name in _ref_list(entry.get("refs")))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_id(component_id: str) -> None:
    if _ID.fullmatch(component_id) is None:
        raise SharedManifestError(f"component id {component_id!r} is not allowed")


def _check_app(app: str) -> None:
    if _APP.fullmatch(app) is None:
        raise SharedManifestError("app id must be a short lowercase name")


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise SharedManifestError(f"component kind {kind!r} is not allowed")


def _check_rel(path: str) -> None:
    if not path or path.startswith(("/", "\\")) or ".." in Path(path).parts:
        raise SharedManifestError("component path must stay inside the shared directory")


def _ref_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: list[str] = []
    for item in value:
        name = item.strip() if isinstance(item, str) else str(item).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _parse(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _needs_migration(data: dict[str, Any]) -> bool:
    schema = data.get("schema")
    if "schema" in data and not _is_int(schema):
        raise SharedManifestError(f"unsupported suite manifest schema {schema}", kind="schema")
    if schema not in (None, SCHEMA):
        raise SharedManifestError(f"unsupported suite manifest schema {schema}", kind="schema")
    if "resources" in data and "components" not in data:
        return True
    components = data.get("components")
    if not isinstance(components, dict):
        return "components" not in data and "resources" in data
    if schema is None:
        return True
    for key, value in components.items():
        if key in {"python", "gpu", "model"}:
            return True
        if not isinstance(value, dict):
            return True
        if "holders" in value or "apps" in value or "kind" not in value or "refs" not in value:
            return True
        if value.get("kind") not in KINDS:
            return True
    return False


def _from_new(data: dict[str, Any]) -> SharedManifest:
    schema = data.get("schema")
    if schema != SCHEMA:
        raise SharedManifestError(f"unsupported suite manifest schema {schema}", kind="schema")
    raw = data.get("components")
    components: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise SharedManifestError("manifest components must be objects")
            _check_id(key)
            kind = value.get("kind")
            if kind not in KINDS:
                raise SharedManifestError(f"component kind {kind!r} is not allowed")
            components[key] = dict(value)
            components[key]["refs"] = _ref_list(value.get("refs"))
    extra = {key: value for key, value in data.items() if key not in {"schema", "components"}}
    return SharedManifest(schema=SCHEMA, components=components, extra=extra)


def _migrate(data: dict[str, Any]) -> SharedManifest:
    extra = {key: value for key, value in data.items() if key not in {"schema", "components", "resources"}}
    if "resources" in data and "components" not in data:
        components = _from_resources(data.get("resources"))
    else:
        raw = data.get("components")
        components = _from_enlarge(raw if isinstance(raw, dict) else {})
    return SharedManifest(schema=SCHEMA, components=components, extra=extra)


def _from_resources(resources: object) -> dict[str, dict[str, Any]]:
    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(resources, dict):
        for key, value in resources.items():
            if isinstance(key, str) and isinstance(value, dict):
                items.append((key, value))
    elif isinstance(resources, list):
        for value in resources:
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                items.append((str(value["id"]), value))
    else:
        raise SharedManifestError("manifest resources must be an object or a list")
    components: dict[str, dict[str, Any]] = {}
    for key, value in items:
        _check_id(key)
        components[key] = _complete(key, value)
    return components


def _from_enlarge(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        entry = dict(value)
        if "refs" not in entry:
            if "holders" in entry:
                entry["refs"] = entry.pop("holders")
            elif "apps" in entry:
                entry["refs"] = entry.pop("apps")
        else:
            entry.pop("holders", None)
            entry.pop("apps", None)
        for new_id, record in _expand(key, entry):
            _check_id(new_id)
            _merge(components, new_id, record)
    return components


def _expand(key: str, entry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    refs = _ref_list(entry.get("refs"))
    if key == "gpu":
        path = str(entry.get("path") or "python")
        pins = (
            ("torch-cu130", "wheels", "cu130", lock_hash(TORCH_LOCK)),
            ("onnxruntime-gpu-1.31", "wheels", "1.31.0", lock_hash(ORT_LOCK)),
            ("tensorrt-10", "wheels", "10.14.1.48", lock_hash(TRT_LOCK)),
        )
        expanded: list[tuple[str, dict[str, Any]]] = []
        for cid, kind, version, digest in pins:
            payload = dict(entry)
            payload.update({"kind": kind, "version": version, "path": path, "sha256": digest, "refs": refs})
            expanded.append((cid, _complete(cid, payload)))
        return expanded
    if key == "python":
        version = str(entry.get("version") or "3.14")
        new_id = f"python-{version}"
        if _ID.fullmatch(new_id) is None:
            new_id = "python-3.14"
            version = "3.14"
        digest = str(entry.get("sha256") or "") or lock_hash(PYTHON_LOCK)
        return [(new_id, _complete(new_id, {**entry, "kind": "runtime", "version": version, "path": str(entry.get("path") or "python"), "sha256": digest, "refs": refs}))]
    if key == "model":
        version = str(entry.get("version") or DEFAULT_MODEL_ID)
        digest = str(entry.get("sha256") or "")
        if not digest and version == DEFAULT_MODEL_ID:
            digest = _default_model_sha()
        prefix = digest[:12] if len(digest) >= 12 else "unknown"
        new_id = f"model-{prefix}"
        path = str(entry.get("path") or f"model/{prefix}")
        return [(new_id, _complete(new_id, {**entry, "kind": "model", "version": version, "path": path, "sha256": digest, "refs": refs}))]
    return [(key, _complete(key, entry))]


def _complete(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    kind = entry.get("kind")
    if kind not in KINDS:
        if key.startswith("python"):
            kind = "runtime"
        elif key.startswith("model"):
            kind = "model"
        elif key.startswith(("torch", "onnxruntime", "tensorrt")):
            kind = "wheels"
        else:
            kind = "tool"
    version = str(entry.get("version") or "")
    path = str(entry.get("path") or f"runtime/{key}")
    _check_rel(path)
    digest = str(entry.get("sha256") or "")
    when = str(entry.get("installed_at") or "") or _now()
    refs = _ref_list(entry.get("refs"))
    unknown = {name: value for name, value in entry.items() if name not in _FIELDS and name not in {"holders", "apps", "id"}}
    record = dict(unknown)
    record.update(
        {
            "installed_at": when,
            "kind": kind,
            "path": path,
            "refs": refs,
            "sha256": digest,
            "version": version,
        }
    )
    return record


def _merge(components: dict[str, dict[str, Any]], component_id: str, record: dict[str, Any]) -> None:
    current = components.get(component_id)
    if current is None:
        components[component_id] = record
        return
    refs = _ref_list(current.get("refs"))
    for name in _ref_list(record.get("refs")):
        if name not in refs:
            refs.append(name)
    current["refs"] = refs
    for name in ("kind", "version", "path", "sha256", "installed_at"):
        if not current.get(name) and record.get(name):
            current[name] = record[name]


def _default_model_sha() -> str:
    model = load_builtin(release=True).get(DEFAULT_MODEL_ID)
    if model is None:
        return ""
    return model.sha256.lower()


def _read_unlocked(folder: Path) -> SharedManifest:
    path = folder / "manifest.json"
    if not path.is_file():
        return SharedManifest()
    data = _parse(path)
    if data is None:
        return SharedManifest()
    if _needs_migration(data):
        return _migrate(data)
    return _from_new(data)


def _write(folder: Path, manifest: SharedManifest) -> None:
    path = folder / "manifest.json"
    tmp = folder / "manifest.json.tmp"
    text = json.dumps(manifest.as_dict(), indent=1, sort_keys=True) + "\n"
    folder.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(folder)


def _fsync_dir(folder: Path) -> None:
    if sys.platform == "win32":
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _locked(folder: Path, timeout: float) -> Iterator[None]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOCK_NAME
    fd = _acquire(path, timeout)
    try:
        yield
    finally:
        _release(fd, path)


def _acquire(path: Path, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
        except FileExistsError:
            if not _lock_busy(path):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise SharedManifestError(f"manifest.lock timed out after {timeout:g}s", kind="lock")
            time.sleep(0.05)
            continue
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        return fd


def _lock_busy(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    first = text.splitlines()[0].strip() if text.strip() else ""
    if not first.isdigit():
        return True
    return _pid_alive(int(first))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _release(fd: int, path: Path) -> None:
    os.close(fd)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    if text.startswith(f"{os.getpid()}\n"):
        try:
            path.unlink()
        except OSError:
            return
