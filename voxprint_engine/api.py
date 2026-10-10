"""Public imports for Clarity. This list is the stable surface.

Additive names can land in a pull request. Removing a name, or changing the
meaning of one, bumps ``API``. The video names in ``__all__`` are that
addition. This package does not import Qt.
"""
from __future__ import annotations

from voxprint_engine import API, __version__
from voxprint_engine.gpu_lock import gpu_job
from voxprint_engine.infer import upscale_file, uses_lanczos
from voxprint_engine.paths import models_dir, trt_cache_dir
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
from voxprint_engine.store import ensure_weight, register, unregister
from voxprint_engine.thermal import decide
from voxprint_engine.video import (
    RestoreBatchRequest,
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
    "decide",
    "ensure_weight",
    "gpu_job",
    "halve_tile",
    "load_builtin",
    "models_dir",
    "note_spill",
    "read_shared_usage",
    "register",
    "restore_frames",
    "sysmem_spill",
    "temporal_windows",
    "tile_frames",
    "trt_cache_dir",
    "unregister",
    "upscale_file",
    "uses_lanczos",
    "validate_manifest",
    "vram_budget_gb",
]
