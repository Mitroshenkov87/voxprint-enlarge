"""Shared model folder and ``<models>/.users.json``.

The file format is the one Audiobook Builder and Movie Dubber already write:
``{"audiobook-builder": true, "movie-dubber": true}``. Enlarge adds the key
``enlarge``. Clarity should add ``clarity``. Unknown keys are kept. A damaged
file is treated as empty and replaced on the next write.

Weights are downloaded only over HTTPS, only from a registry host, and only
when the manifest pins a sha256. A redirect must stay on that same registry.
Private and link-local addresses are refused. Non-commercial and unlicensed
manifests are refused before any byte is fetched.
"""
from __future__ import annotations

import ipaddress
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from http.client import HTTPMessage
from pathlib import Path
from typing import IO

from voxprint_common.download import DownloadError, download as fetch
from voxprint_common.hashing import sha256_file as hash_file
from voxprint_common.licence import is_noncommercial, is_unlicensed
from voxprint_common.paths import models_dir
from voxprint_engine.registry.schema import ModelManifest

USERS_FILE = ".users.json"
ENLARGE_USER = "enlarge"


class StoreError(RuntimeError):
    """The weight file could not be resolved, downloaded, or checked."""


# Hosts the catalogue is allowed to download from, grouped by registry.
# GitHub release files and Hugging Face LFS files are served from those
# registries' own asset hosts. A redirect to any other host is refused.
_REGISTRY_GROUPS: dict[str, str] = {
    "github.com": "github",
    "objects.githubusercontent.com": "github",
    "release-assets.githubusercontent.com": "github",
    "github-releases.githubusercontent.com": "github",
    "huggingface.co": "huggingface",
    "cdn-lfs.huggingface.co": "huggingface",
    "cdn-lfs-us-1.huggingface.co": "huggingface",
    "cdn-lfs-eu-1.huggingface.co": "huggingface",
    "cas-bridge.xethub.hf.co": "huggingface",
    "kaggle.com": "kaggle",
    "www.kaggle.com": "kaggle",
}
Resolver = Callable[[str], list[str]]


def registry_group(host: str) -> str | None:
    """The registry that owns ``host``, or ``None`` when it is not allowed."""
    return _REGISTRY_GROUPS.get(host.lower().rstrip("."))


def _addresses(host: str, resolve: Resolver | None = None) -> list[str]:
    """IP addresses for ``host``. ``resolve`` replaces DNS in tests."""
    if resolve is not None:
        return list(resolve(host))
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise StoreError(f"could not resolve {host}") from exc
    found: list[str] = []
    for info in infos:
        sockaddr = info[4]
        raw = sockaddr[0] if sockaddr else ""
        if isinstance(raw, str) and raw and raw not in found:
            found.append(raw)
    return found


def _blocked_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def require_registry_url(url: str, *, resolve: Resolver | None = None) -> str:
    """Refuse a URL that is not HTTPS on one registry host with public addresses.

    Args:
        url: Candidate download URL.
        resolve: Optional DNS replacement. Production uses the system resolver.

    Returns:
        The hostname, lowercased.

    Raises:
        StoreError: The scheme, host, or an address is not acceptable.
    """
    parsed = urllib.parse.urlsplit(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not host:
        raise StoreError("downloads must use https")
    if parsed.username or parsed.password:
        raise StoreError("downloads must not include credentials")
    group = registry_group(host)
    if group is None:
        raise StoreError(f"downloads must come from a registry host, not {host}")
    for address in _addresses(host, resolve):
        if _blocked_address(address):
            raise StoreError(f"downloads must not use a private address for {host}")
    return host


class _SameRegistryRedirect(urllib.request.HTTPRedirectHandler):
    """Follow an HTTPS redirect only when it stays on the same registry."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Reject a redirect that leaves the registry or uses a private address."""
        origin = require_registry_url(req.full_url)
        target = (urllib.parse.urlsplit(newurl).hostname or "").lower().rstrip(".")
        if registry_group(origin) != registry_group(target):
            raise StoreError("downloads must not redirect to another host")
        require_registry_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_registry(request: urllib.request.Request, timeout: float) -> object:
    """Open one registry URL, refusing a redirect off that registry."""
    require_registry_url(request.full_url)
    opener = urllib.request.build_opener(_SameRegistryRedirect)
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except StoreError:
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise StoreError(str(exc)) from exc


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

    The URL must be HTTPS on a Hugging Face, Kaggle, or GitHub release host.
    A redirect off that registry is refused, including a redirect to a private
    address. The shared downloader then checks the digest, resumes a partial
    file, and refuses the transfer when the disk is short.
    """
    require_registry_url(url)

    def _progress(notice: object) -> None:
        if on_progress is None:
            return
        fraction = getattr(notice, "fraction", None)
        if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
            on_progress(float(fraction), f"downloading {target.name}")

    try:
        fetch(target, [url], sha256=expected_sha256, progress=_progress, opener=_open_registry)
    except DownloadError as exc:
        raise StoreError(str(exc)) from exc
    if on_progress is not None:
        on_progress(1.0, f"saved {target.name}")
