"""Shared image-upscale engine for the Voxprint suite.

Voxprint AI Enlarge owns this package. Voxprint AI Clarity imports it
(registry, worker IPC, ONNX tiling, model store, GPU lock, VRAM budget,
thermal rule) and co-maintains it. The package does not import Qt and does
not start a process on import. Runtime behaviour is documented in
``docs/ENGINE.md``.

``API`` is the compatibility number. A breaking change to the manifest schema
or the worker JSON lines bumps it.
"""
from __future__ import annotations

API = 1
__version__ = "0.1.0"

__all__ = ["API", "__version__"]
