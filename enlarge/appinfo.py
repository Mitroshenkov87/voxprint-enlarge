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


def version_label() -> str:
    return format_version()


def resource_dir() -> Path:
    """Folder with bundled resources: the repository root, or the frozen bundle."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def build_info() -> dict[str, Any]:
    try:
        payload = json.loads((resource_dir() / "build_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def version_line() -> str:
    info = build_info()
    parts: list[str] = []
    if info.get("commit"):
        parts.append(f"commit {info['commit']}")
    if info.get("built"):
        parts.append(f"built {info['built']}")
    parts.append("frozen exe" if getattr(sys, "frozen", False) else "run from source")
    return f"{version_label()} ({', '.join(parts)})"
