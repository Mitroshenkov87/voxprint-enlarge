"""Application metadata. The codename stays empty until a release build is named."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

APP_NAME = "VoxprintEnlarge"
APP_DISPLAY_NAME = "Voxprint AI Enlarge"
APP_VERSION = "0.1.0-pre"
APP_BUILD = 0
CODENAME = ""
STATUS = "pre-alpha"
START_MENU_FOLDER = "Voxprint"


def format_version(version: str = APP_VERSION, build: int = APP_BUILD, codename: str = CODENAME) -> str:
    """Human version, for example ``0.1.0 pre-alpha · build 0``."""
    pretty = version[:-4] + " pre-alpha" if version.endswith("-pre") else version
    label = f"{pretty} · build {int(build)}"
    name = (codename or "").strip()
    if name:
        label += f' "{name}"'
    return label


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def catalog_info() -> dict[str, Any]:
    """``BUILD.json`` at the repository root. A release stamp does not replace this file."""
    return _json_object(resource_dir() / "BUILD.json")


def product_version() -> str:
    """Version ``--version`` and ``--capabilities`` both print.

    A stamped ``build_info.json`` wins, then ``BUILD.json``, then ``APP_VERSION``.
    """
    stamped = str(build_info().get("version") or "").strip()
    if stamped:
        return stamped
    catalog = str(catalog_info().get("version") or "").strip()
    if catalog:
        return catalog
    return APP_VERSION


def status_label() -> str:
    """Short status next to the version in the window."""
    version = product_version()
    if version.endswith("-pre"):
        return "pre-alpha"
    if "-rc" in version:
        return "release candidate"
    if "-alpha" in version:
        return "alpha"
    return STATUS


def version_label() -> str:
    """Human version line, including the build number and codename when they are set."""
    stamped = build_info()
    catalog = catalog_info()
    version = product_version()
    raw_build = stamped.get("build", APP_BUILD)
    try:
        build = int(raw_build)
    except (TypeError, ValueError):
        build = APP_BUILD
    codename = str(stamped.get("codename") or catalog.get("codename") or CODENAME)
    return format_version(version, build, codename)


def resource_dir() -> Path:
    """Folder with bundled resources: the repository root, or the frozen bundle."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def build_info() -> dict[str, Any]:
    """Stamped ``build_info.json`` from a release bundle. Empty in a source checkout."""
    return _json_object(resource_dir() / "build_info.json")


def version_line() -> str:
    """``--version`` text: the human version plus commit, build time, and source or frozen."""
    info = build_info()
    parts: list[str] = []
    if info.get("commit"):
        parts.append(f"commit {info['commit']}")
    if info.get("built"):
        parts.append(f"built {info['built']}")
    if getattr(sys, "frozen", False):
        origin = "frozen exe"
    elif info.get("commit"):
        origin = "installed build"
    else:
        origin = "run from source"
    parts.append(origin)
    return f"{version_label()} ({', '.join(parts)})"
