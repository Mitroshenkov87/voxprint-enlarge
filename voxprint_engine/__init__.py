"""Shared image-upscale engine for the Voxprint suite.

Voxprint AI Enlarge owns this package. Voxprint AI Clarity imports it
(registry, worker IPC, ONNX tiling, the model store, and the VRAM budget)
and co-maintains it. Paths, the GPU lock, thermal decisions, and the
runtime manifest live in ``voxprint_common``. The package does not import Qt
and does not start a process on import. Runtime behaviour is documented in
``docs/ENGINE.md``.

``API`` is the compatibility number. A breaking change to the worker JSON
lines bumps it. ``voxprint_common.API`` is a separate number.
"""
from __future__ import annotations

API = 1
__version__ = "1.4.0"

__all__ = ["API", "__version__"]
