"""Load the built-in catalogue and optional custom manifest files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.registry.validate import ManifestError, validate_manifest

_BUILTIN = Path(__file__).resolve().parent / "catalog.json"


@dataclass
class Catalog:
    """Models in display order, plus warnings from files that were skipped."""

    models: list[ModelManifest] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def get(self, model_id: str) -> ModelManifest | None:
        """The manifest for this id, or None."""
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def selectable(self) -> list[ModelManifest]:
        """Models the user can pick. Lanczos and face entries stay out."""
        return [model for model in self.models if model.selectable]

    def credits(self) -> list[ModelManifest]:
        """CC-BY models, in catalogue order, for the credits screen."""
        return [model for model in self.models if model.requires_credit()]


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _models_in(payload: object) -> list[object]:
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        return list(payload["models"])
    if isinstance(payload, dict) and "id" in payload:
        return [payload]
    if isinstance(payload, list):
        return list(payload)
    raise ManifestError(["file must be a manifest, a list of manifests, or {\"models\": [...]}"])


def load_builtin(*, release: bool = True) -> Catalog:
    """The catalogue shipped with the engine. A failure here is a packaging bug."""
    payload = _read_json(_BUILTIN)
    models: list[ModelManifest] = []
    for item in _models_in(payload):
        if not isinstance(item, dict):
            raise ManifestError(["built-in catalogue item is not an object"])
        models.append(validate_manifest(item, release=release))
    ids = [model.id for model in models]
    if len(ids) != len(set(ids)):
        raise ManifestError(["built-in catalogue has a duplicate id"])
    return Catalog(models=models)


def _load_custom_file(path: Path, *, release: bool) -> tuple[list[ModelManifest], list[str]]:
    try:
        payload = _read_json(path)
        rows = _models_in(payload)
    except (OSError, ValueError, ManifestError) as exc:
        return [], [f"{path.name}: {exc}"]
    models: list[ModelManifest] = []
    warnings: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            warnings.append(f"{path.name}: item {index} is not an object")
            continue
        try:
            models.append(validate_manifest(row, release=release))
        except ManifestError as exc:
            warnings.append(f"{path.name}: item {index}: {exc}")
    return models, warnings


def combined_catalog(custom_dir: Path | None = None, *, release: bool = True) -> Catalog:
    """Built-in models, then any valid manifests in ``custom_dir``.

    A custom file that repeats a built-in id is skipped. Invalid files are
    reported in ``warnings`` and do not drop the built-in models.
    """
    base = load_builtin(release=release)
    if custom_dir is None:
        return base
    folder = Path(custom_dir)
    if not folder.is_dir():
        base.warnings.append(f"custom models folder is not a directory: {folder}")
        return base
    seen = {model.id for model in base.models}
    extra: list[ModelManifest] = []
    warnings = list(base.warnings)
    for path in sorted(folder.glob("*.json")):
        found, problems = _load_custom_file(path, release=release)
        warnings.extend(problems)
        for model in found:
            if model.id in seen:
                warnings.append(f"{path.name}: duplicate id {model.id} skipped")
                continue
            seen.add(model.id)
            extra.append(model)
    return Catalog(models=[*base.models, *extra], warnings=warnings)
