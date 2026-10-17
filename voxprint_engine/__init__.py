"""Shared image-upscale engine for the VOXPrint suite.

VOXPrint AI Enlarge owns this package. VOXPrint AI Clarity imports it
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
__version__ = "1.5.2"

__all__ = ["API", "__version__", "prepare_native_runtime", "session_providers"]


def __getattr__(name: str) -> object:
    """Load the native-runtime helpers on first use.

    Importing them from this module body would import onnxruntime. Callers
    that only need ``API`` and ``__version__`` stay free of that import.
    """
    if name == "prepare_native_runtime":
        from voxprint_engine.trt_libs import prepare_native_runtime

        return prepare_native_runtime
    if name == "session_providers":
        from voxprint_engine.session import session_providers

        return session_providers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
