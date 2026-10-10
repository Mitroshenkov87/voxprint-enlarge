"""Credits text built from the model registry. Names are not hardcoded here."""
from __future__ import annotations

from voxprint_engine.registry.schema import ModelManifest


def credit_sections(models: list[ModelManifest]) -> list[tuple[str, list[ModelManifest]]]:
    """Group models by licence. CC-BY sections come first."""
    grouped: dict[str, list[ModelManifest]] = {}
    for model in models:
        grouped.setdefault(model.licence, []).append(model)
    order = sorted(grouped, key=lambda licence: (not licence.startswith("CC-BY"), licence))
    return [(licence, grouped[licence]) for licence in order]
