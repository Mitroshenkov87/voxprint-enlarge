"""Enlarge-only settings. Shared GPU and models folder live in ``suite.json``."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from enlarge.paths import app_home
from voxprint_engine.formats import DEFAULT_FORMAT, DEFAULT_QUALITY, normalize_format
from voxprint_engine.registry import DEFAULT_MODEL_ID

DEFAULTS: dict[str, Any] = {
    "model_id": DEFAULT_MODEL_ID,
    "scale": 4,
    "format": DEFAULT_FORMAT,
    "quality": DEFAULT_QUALITY,
    "lossless": False,
    "double_upscale": False,
    "tile": "auto",
    "output_dir": "",
    "custom_models_dir": "",
    "include_subfolders": True,
}
TILES = ("auto", "128", "256", "512", "768")


def path() -> Path:
    """Path of this app's settings file."""
    return app_home() / "settings.json"


def _clean(raw: object) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    data = dict(DEFAULTS)
    if isinstance(source.get("model_id"), str) and source["model_id"].strip():
        data["model_id"] = source["model_id"].strip()
    if source.get("scale") in (2, 4):
        data["scale"] = source["scale"]
    data["format"] = normalize_format(source.get("format"))
    quality = source.get("quality")
    if isinstance(quality, int) and not isinstance(quality, bool) and 1 <= quality <= 100:
        data["quality"] = quality
    if isinstance(source.get("lossless"), bool):
        data["lossless"] = source["lossless"]
    if data["lossless"] and data["format"] != "jxl":
        data["lossless"] = False
    if isinstance(source.get("double_upscale"), bool):
        data["double_upscale"] = source["double_upscale"]
    tile = source.get("tile")
    if isinstance(tile, str) and tile in TILES:
        data["tile"] = tile
    if isinstance(source.get("output_dir"), str):
        data["output_dir"] = source["output_dir"].strip()
    if isinstance(source.get("custom_models_dir"), str):
        data["custom_models_dir"] = source["custom_models_dir"].strip()
    if isinstance(source.get("include_subfolders"), bool):
        data["include_subfolders"] = source["include_subfolders"]
    return data


def load() -> dict[str, Any]:
    """Read settings, filling any missing key from the defaults."""
    try:
        raw = json.loads(path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raw = {}
    return _clean(raw)


def save(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into the settings file and return the stored mapping."""
    cleaned = _clean({**load(), **updates})
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(cleaned, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    try:
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return cleaned
