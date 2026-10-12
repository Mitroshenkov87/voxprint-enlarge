"""Public imports for Clarity. This list is the stable surface.

Additive names can land in a pull request. Removing a name, or changing the
meaning of one, bumps ``API``. 1.4.0 removes the suite helpers that now live
in ``voxprint_common`` (paths, the GPU lock, the thermal rule, and the
runtime installer). 1.5.0 adds ``prepare_native_runtime`` and
``session_providers``. ``API`` stays 1: the new names are additive, and
``voxprint-common`` 0.1 does not require an engine API bump. This package
does not import Qt.
"""
from __future__ import annotations

from voxprint_engine import API, __version__
from voxprint_engine.infer import upscale_file, uses_lanczos
from voxprint_engine.onnx_export import ensure_onnx
from voxprint_engine.registry import (
    DEFAULT_MODEL_ID,
    LANCZOS_MODEL_ID,
    Catalog,
    ManifestError,
    ModelManifest,
    combined_catalog,
    load_builtin,
    validate_manifest,
)
from voxprint_engine.session import session_providers
from voxprint_engine.store import ensure_weight, register, unregister
from voxprint_engine.trt_libs import prepare_native_runtime
from voxprint_engine.video import (
    RestoreBatchRequest,
    iter_tile_frames,
    read_shared_usage,
    restore_frames,
    temporal_windows,
    tile_frames,
)
from voxprint_engine.vram import choose_tile, halve_tile, note_spill, sysmem_spill, vram_budget_gb
from voxprint_engine.worker import UpscaleRequest, UpscaleResult, WorkerClient, WorkerError

__all__ = [
    "API",
    "Catalog",
    "DEFAULT_MODEL_ID",
    "LANCZOS_MODEL_ID",
    "ManifestError",
    "ModelManifest",
    "RestoreBatchRequest",
    "UpscaleRequest",
    "UpscaleResult",
    "WorkerClient",
    "WorkerError",
    "__version__",
    "choose_tile",
    "combined_catalog",
    "ensure_onnx",
    "ensure_weight",
    "halve_tile",
    "iter_tile_frames",
    "load_builtin",
    "note_spill",
    "prepare_native_runtime",
    "read_shared_usage",
    "register",
    "restore_frames",
    "session_providers",
    "sysmem_spill",
    "temporal_windows",
    "tile_frames",
    "unregister",
    "upscale_file",
    "uses_lanczos",
    "validate_manifest",
    "vram_budget_gb",
]
