"""Shared ``suite.json`` (schema 1), the same file Audiobook Builder and Movie Dubber use.

Enlarge reads and writes ``gpu`` and ``models_dir``. It leaves ``ui_language`` and
``theme`` alone, including values it does not use, so opening Enlarge does not
reset another app's language. Unknown keys are preserved.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from enlarge.paths import voxprint_home

SCHEMA = 1
FILENAME = "suite.json"
DEFAULT_GPU = "auto"
_GPU_RE = re.compile(r"^cuda:(\d{1,2})$")


def path() -> Path:
    return voxprint_home() / "state" / FILENAME


def read_raw() -> dict[str, Any]:
    try:
        data = json.loads(path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def normalize_gpu(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in ("auto", "cpu"):
        return text
    if text in ("cuda", "gpu"):
        return "cuda:0"
    match = _GPU_RE.match(text)
    if match is None:
        return None
    return f"cuda:{int(match.group(1))}"


def gpu() -> str:
    value = read_raw().get("gpu")
    if isinstance(value, dict):
        value = value.get("device", value.get("gpu"))
    return normalize_gpu(value) or DEFAULT_GPU


def models_dir() -> Path | None:
    """Absolute folder from the file, or None when the key is missing, null, or relative."""
    value = read_raw().get("models_dir")
    if not isinstance(value, str) or not value.strip():
        return None
    folder = Path(os.path.expandvars(os.path.expanduser(value.strip().strip('"'))))
    return folder if folder.is_absolute() else None


def resolved_models_dir() -> Path:
    """``VOXPRINT_MODELS_DIR``, else the suite value, else the default shared folder."""
    env = os.environ.get("VOXPRINT_MODELS_DIR", "").strip().strip('"')
    if env:
        folder = Path(os.path.expandvars(os.path.expanduser(env)))
        if folder.is_absolute():
            return folder
    return models_dir() or (voxprint_home() / "models")


def _write(data: dict[str, Any]) -> None:
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    try:
        for attempt in range(10):
            try:
                os.replace(tmp, target)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        raise PermissionError(f"cannot replace {target}")
    finally:
        if tmp.exists():
            tmp.unlink()


def set_gpu(value: str) -> str:
    stored = normalize_gpu(value)
    if stored is None:
        raise ValueError(f"gpu must be auto, cpu, or cuda:N, got {value!r}")
    data = read_raw()
    data["schema"] = SCHEMA
    data["gpu"] = stored
    _write(data)
    return stored


def set_models_dir(folder: Path | None) -> None:
    """None stores JSON null, which means the default shared folder."""
    data = read_raw()
    data["schema"] = SCHEMA
    if folder is None:
        data["models_dir"] = None
    else:
        if not folder.is_absolute():
            raise ValueError("models_dir must be an absolute path")
        data["models_dir"] = str(folder)
    _write(data)
