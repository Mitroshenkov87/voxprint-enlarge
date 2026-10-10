"""Model registry: manifest schema, licence gate, and the built-in catalogue.

Clarity imports these names. Weights are not stored here; a manifest describes
a model and its licence so a release build can refuse it before anything loads.
"""
from __future__ import annotations

from voxprint_engine.registry.catalog import Catalog, combined_catalog, load_builtin
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.registry.validate import ManifestError, validate_manifest

DEFAULT_MODEL_ID = "4x-nomos2-realplksr-dysample"
LANCZOS_MODEL_ID = "lanczos"

__all__ = [
    "DEFAULT_MODEL_ID",
    "LANCZOS_MODEL_ID",
    "Catalog",
    "ManifestError",
    "ModelManifest",
    "combined_catalog",
    "load_builtin",
    "validate_manifest",
]
