"""The one preset-to-registry map.

``presets/mdl-map.json`` is the only place those registry ids are written.
Model Forge replaces the ``model_id`` values later. The preset names stay.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voxprint_common.ui.presets import PresetFile, parse_presets
from voxprint_common.ui.settings import Settings

_SCHEMA = "enlarge-mdl-map/1"


@dataclass(frozen=True)
class PicturePreset:
    """One plain-word picture choice and the registry id it writes."""

    option_id: str
    label_key: str
    ipc_preset: str
    ipc_qualities: tuple[str, ...]
    model_id: str


def map_path() -> Path:
    """Return the placeholder mapping file."""
    return Path(__file__).resolve().parent / "presets" / "mdl-map.json"


def picture_map(path: Path | None = None) -> tuple[PicturePreset, ...]:
    """Return the picture presets in file order.

    Args:
        path: Mapping file. The default is :func:`map_path`.

    Raises:
        ValueError: The file is missing a picture row or uses another schema.
    """
    source = map_path() if path is None else path
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"preset map {source} is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError(f"preset map schema must be {_SCHEMA}")
    raw_rows = payload.get("pictures")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("preset map needs picture rows")
    rows: list[PicturePreset] = []
    seen: set[str] = set()
    for raw in raw_rows:
        row = _picture(raw)
        if row.option_id in seen:
            raise ValueError(f"preset map repeats {row.option_id}")
        seen.add(row.option_id)
        rows.append(row)
    return tuple(rows)


def model_id_for(ipc_preset: str, ipc_quality: str, path: Path | None = None) -> str:
    """Return the registry id for an IPC preset and quality, or empty text.

    Args:
        ipc_preset: ``photo`` or ``art``.
        ipc_quality: ``fast`` or ``max``.
        path: Mapping file. The default is :func:`map_path`.
    """
    for row in picture_map(path):
        if row.ipc_preset == ipc_preset and ipc_quality in row.ipc_qualities:
            return row.model_id
    return ""


def load_enlarge_presets(settings: Settings, path: Path | None = None) -> PresetFile:
    """Load the preset document and fill model ids from the map.

    Args:
        settings: Descriptors the values are checked against.
        path: Preset document. The default sits beside the map.

    Raises:
        ValueError: A picture chip has no row in the map.
    """
    source = Path(__file__).resolve().parent / "presets" / "enlarge.json" if path is None else path
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"presets file {source} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"presets file {source} is not an object")
    by_id = {row.option_id: row.model_id for row in picture_map()}
    groups = payload.get("groups")
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict) or group.get("id") != "pictures":
                continue
            options = group.get("options")
            if not isinstance(options, list):
                continue
            for option in options:
                if not isinstance(option, dict):
                    continue
                option_id = option.get("id")
                model_id = by_id.get(option_id) if isinstance(option_id, str) else None
                if not model_id:
                    raise ValueError(f"mdl-map.json has no picture preset {option_id}")
                values = option.get("values")
                if not isinstance(values, dict):
                    values = {}
                    option["values"] = values
                values["enlarge.model"] = model_id
    return parse_presets(payload, settings)


def _picture(raw: object) -> PicturePreset:
    if not isinstance(raw, dict):
        raise ValueError("a picture preset must be an object")
    option_id = raw.get("id")
    label_key = raw.get("label_key")
    ipc_preset = raw.get("ipc_preset")
    qualities = raw.get("ipc_qualities")
    model_id = raw.get("model_id")
    if (
        not isinstance(option_id, str)
        or not option_id
        or not isinstance(label_key, str)
        or not label_key
        or not isinstance(ipc_preset, str)
        or not ipc_preset
        or not isinstance(model_id, str)
        or not model_id
        or not isinstance(qualities, list)
        or not qualities
        or not all(isinstance(item, str) and item for item in qualities)
    ):
        raise ValueError("a picture preset needs an id, a label, an IPC preset, qualities, and a model id")
    return PicturePreset(option_id, label_key, ipc_preset, tuple(qualities), model_id)
