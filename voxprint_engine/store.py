"""Shared model folder and ``<models>/.users.json``.

The file format is the one Audiobook Builder and Movie Dubber already write:
``{"audiobook-builder": true, "movie-dubber": true}``. Enlarge adds the key
``enlarge``. Clarity should add ``clarity``. Unknown keys are kept. A damaged
file is treated as empty and replaced on the next write.

Weights are downloaded only over HTTPS and only when the manifest pins a
sha256. Non-commercial and unlicensed manifests are refused before any byte
is fetched.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from voxprint_common.download import DownloadError, download as fetch
from voxprint_common.hashing import sha256_file as hash_file
from voxprint_common.licence import is_noncommercial, is_unlicensed
from voxprint_common.paths import models_dir
from voxprint_engine.registry.schema import ModelManifest

USERS_FILE = ".users.json"
ENLARGE_USER = "enlarge"


class StoreError(RuntimeError):
    """The weight file could not be resolved, downloaded, or checked."""


def users_path(root: Path | None = None) -> Path:
    """Path of .users.json beside the shared weights."""
    return (root or models_dir()) / USERS_FILE


def read_users(root: Path | None = None) -> dict[str, bool]:
    """``{program: true/false}``. Missing or unreadable files are empty."""
    try:
        data = json.loads(users_path(root).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): bool(value) for key, value in data.items()}


def _write_users(users: dict[str, bool], root: Path | None = None) -> None:
    path = users_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(users, indent=1, sort_keys=True), encoding="utf-8")
    try:
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        raise StoreError(f"cannot replace {path}")
    finally:
        if tmp.exists():
            tmp.unlink()


def register(key: str, root: Path | None = None) -> dict[str, bool]:
    """Add ``key`` when it is not already true. Other programs' keys stay."""
    if not key.strip():
        raise StoreError("a models user key is required")
    users = read_users(root)
    if users.get(key) is True:
        return users
    users[key] = True
    _write_users(users, root)
    return users


def unregister(key: str, root: Path | None = None) -> list[str]:
    """Remove ``key``. Returns the other programs that are still true."""
    users = read_users(root)
    if key in users:
        users.pop(key)
        _write_users(users, root)
    return sorted(name for name, present in users.items() if present)


def register_quietly(key: str = ENLARGE_USER, root: Path | None = None) -> None:
    """App start. A read-only models folder must not stop the program."""
    try:
        register(key, root)
    except (OSError, StoreError):
        return


def weight_report(model: ModelManifest, root: Path | None = None) -> tuple[str, str]:
    """``missing``, ``ready``, ``mismatch``, or ``unpinned``, plus a short detail."""
    if not model.sha256 or not model.filename:
        return "unpinned", "this model has no pinned file"
    target = (root or models_dir()) / model.filename
    if not target.is_file():
        return "missing", "not downloaded"
    found = hash_file(target)
    if found != model.sha256.lower():
        return "mismatch", "the file does not match the pinned sha256"
    return "ready", "sha256 matches"


def delete_weight(model: ModelManifest, root: Path | None = None, *, user: str = ENLARGE_USER) -> None:
    """Delete one weight file when no other suite program still lists itself in ``.users.json``."""
    others = sorted(name for name, present in read_users(root).items() if present and name != user)
    if others:
        raise StoreError("still used by " + ", ".join(others))
    if not model.filename or "/" in model.filename or "\\" in model.filename:
        raise StoreError(f"{model.id} has an unsafe filename")
    target = (root or models_dir()) / model.filename
    target.unlink(missing_ok=True)
    target.with_name(target.name + ".partial").unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Lowercase sha256 of a file. The implementation is ``voxprint_common.hashing``."""
    return hash_file(path)


def _reject_licence(model: ModelManifest) -> None:
    if is_unlicensed(model.licence) or is_noncommercial(model.licence):
        raise StoreError(f"{model.id} is not licensed for a release build ({model.licence})")


def ensure_weight(
    model: ModelManifest,
    root: Path | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Return the pinned weight file, downloading it when it is not there yet."""
    _reject_licence(model)
    if not model.download_url or not model.sha256:
        raise StoreError(f"{model.id} has no pinned download")
    if not model.filename or "/" in model.filename or "\\" in model.filename:
        raise StoreError(f"{model.id} has an unsafe filename")
    folder = root or models_dir()
    target = folder / model.filename
    if target.is_file():
        found = hash_file(target)
        if found != model.sha256:
            raise StoreError(f"{target.name} does not match the pinned sha256")
        return target
    download(model.download_url, target, model.sha256, on_progress)
    return target


def download(
    url: str,
    target: Path,
    expected_sha256: str,
    on_progress: Callable[[float, str], None] | None = None,
) -> None:
    """Stream ``url`` to ``target`` through ``voxprint_common.download``.

    Model weights stay on HTTPS. The shared downloader checks the digest,
    resumes a partial file, and refuses the transfer when the disk is short.
    """
    if not url.startswith("https://"):
        raise StoreError("downloads must use https")

    def _progress(notice: object) -> None:
        if on_progress is None:
            return
        fraction = getattr(notice, "fraction", None)
        if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
            on_progress(float(fraction), f"downloading {target.name}")

    try:
        fetch(target, [url], sha256=expected_sha256, progress=_progress)
    except DownloadError as exc:
        raise StoreError(str(exc)) from exc
    if on_progress is not None:
        on_progress(1.0, f"saved {target.name}")
