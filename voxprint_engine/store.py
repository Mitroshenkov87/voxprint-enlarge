"""Shared model files and per-model refs in ``<models>/.users.json``.

VOXPrint MDL downloads weights. This module only locates a file MDL already
placed at ``<models>/<filename>`` or ``<models>/blobs/<sha256>``, and records
which app uses that model. It does not register or delete a shared root
(``models``, ``runtime``, ``voices``, ``cache``, or ``blobs``).

Unknown keys in ``.users.json`` are kept. A legacy ``{key: true}`` still counts
as a holder. Refs are one model id at a time.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from voxprint_common.hashing import sha256_file as hash_file
from voxprint_common.licence import is_noncommercial, is_unlicensed
from voxprint_common.model_users import UsersError, add_user, load_users, remove_user, users_file
from voxprint_common.paths import models_dir
from voxprint_engine.registry.schema import ModelManifest

ENLARGE_USER = "enlarge"
MDL_COMMAND = "voxprint-mdl download --app enlarge"
_SHARED_ROOTS = frozenset({"models", "runtime", "voices", "cache", "blobs"})


class StoreError(RuntimeError):
    """The weight file could not be resolved or checked."""


def mdl_hint(model_id: str = "") -> str:
    """Tell a headless caller to download with VOXPrint MDL."""
    if model_id:
        return f"{model_id} is not downloaded. Run {MDL_COMMAND}"
    return f"Run {MDL_COMMAND}"


def users_path(root: Path | None = None) -> Path:
    """Path of .users.json beside the shared weights."""
    return users_file(root)


def register(model_id: str, root: Path | None = None, *, app: str = ENLARGE_USER) -> dict[str, object]:
    """Record ``app`` on ``model_id``. Other keys stay. A shared root is refused."""
    key = _model_key(model_id)
    try:
        add_user(key, app, path=users_file(root))
        return load_users(users_file(root))
    except UsersError as exc:
        raise StoreError(str(exc)) from exc


def unregister(model_id: str, root: Path | None = None, *, app: str = ENLARGE_USER) -> list[str]:
    """Drop ``app`` from ``model_id``. Returns the other apps that still hold it."""
    key = _model_key(model_id)
    path = users_file(root)
    try:
        remove_user(key, app, path=path)
        document = load_users(path)
    except UsersError as exc:
        raise StoreError(str(exc)) from exc
    return _holders(document.get(key), exclude=app)


def register_quietly(model_id: str = "", root: Path | None = None, *, app: str = ENLARGE_USER) -> None:
    """Record one model when ``model_id`` is set. An empty id writes nothing."""
    if not model_id.strip():
        return
    try:
        register(model_id, root, app=app)
    except (OSError, StoreError):
        return


def weight_report(model: ModelManifest, root: Path | None = None) -> tuple[str, str]:
    """``missing``, ``ready``, ``mismatch``, or ``unpinned``, plus a short detail."""
    if not model.sha256 or not model.filename:
        return "unpinned", "this model has no pinned file"
    target = _located_weight(model, root or models_dir())
    if target is None:
        return "missing", mdl_hint(model.id)
    found = hash_file(target)
    if found != model.sha256.lower():
        return "mismatch", "the file does not match the pinned sha256"
    return "ready", "sha256 matches"


def delete_weight(model: ModelManifest, root: Path | None = None, *, user: str = ENLARGE_USER) -> None:
    """Delete one model's files when no other app still holds that model id."""
    folder = root or models_dir()
    try:
        document = load_users(users_file(folder))
    except UsersError as exc:
        raise StoreError(str(exc)) from exc
    holders = _holders(document.get(model.id), exclude=user)
    if holders:
        raise StoreError("still used by " + ", ".join(holders))
    unregister(model.id, folder, app=user)
    for target in _owned_files(model, folder):
        if _is_shared_root(target, folder):
            raise StoreError(f"refusing to delete a shared root for {model.id}")
        if target.is_file():
            target.unlink()
        partial = target.with_name(target.name + ".partial")
        if partial.is_file():
            partial.unlink()


def sha256_file(path: Path) -> str:
    """Lowercase sha256 of a file. The implementation is ``voxprint_common.hashing``."""
    return hash_file(path)


def ensure_weight(
    model: ModelManifest,
    root: Path | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Return the pinned weight file. VOXPrint MDL is what downloads it."""
    del on_progress
    _reject_licence(model)
    if not model.sha256:
        raise StoreError(mdl_hint(model.id))
    if model.filename and ("/" in model.filename or "\\" in model.filename):
        raise StoreError(f"{model.id} has an unsafe filename")
    folder = root or models_dir()
    target = _located_weight(model, folder)
    if target is None:
        raise StoreError(mdl_hint(model.id))
    found = hash_file(target)
    if found != model.sha256.lower():
        raise StoreError(f"{target.name} does not match the pinned sha256")
    return target


def _reject_licence(model: ModelManifest) -> None:
    if is_unlicensed(model.licence) or is_noncommercial(model.licence):
        raise StoreError(f"{model.id} is not licensed for a release build ({model.licence})")


def _model_key(model_id: str) -> str:
    key = model_id.strip()
    if not key or key in _SHARED_ROOTS:
        raise StoreError("refs are per model, not a shared root")
    return key


def _located_weight(model: ModelManifest, folder: Path) -> Path | None:
    for candidate in _weight_files(model, folder):
        if candidate.is_file():
            return candidate
    return None


def _weight_files(model: ModelManifest, folder: Path) -> tuple[Path, ...]:
    found: list[Path] = []
    filename = model.filename.strip()
    if filename and "/" not in filename and "\\" not in filename:
        found.append(folder / filename)
    digest = model.sha256.strip().lower()
    if len(digest) == 64:
        found.append(folder / "blobs" / digest)
    return tuple(found)


def _owned_files(model: ModelManifest, folder: Path) -> tuple[Path, ...]:
    found = list(_weight_files(model, folder))
    digest = model.sha256.strip().lower()
    if len(digest) == 64:
        found.append(folder / f"{digest}.onnx")
    if model.id and "/" not in model.id and "\\" not in model.id:
        found.append(folder / "enlarge" / model.id / "model.onnx")
    return tuple(found)


def _is_shared_root(path: Path, folder: Path) -> bool:
    try:
        resolved = path.resolve()
        roots = {
            folder.resolve(),
            (folder / "blobs").resolve(),
        }
    except OSError:
        return True
    return resolved in roots


def _holders(value: object, *, exclude: str) -> list[str]:
    names: list[str] = []
    if isinstance(value, list):
        names = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    elif isinstance(value, Mapping):
        refs = value.get("refs")
        if isinstance(refs, list):
            names = [item.strip() for item in refs if isinstance(item, str) and item.strip()]
        if value.get("legacy") is True and not names:
            return ["legacy"]
    elif value is True:
        return ["legacy"]
    return sorted({name for name in names if name != exclude})
