"""ONNX Runtime session. One provider at a time, so a failed TensorRT build can fall through."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from voxprint_engine.providers import CPU, ProviderChoice


class EngineBuildError(RuntimeError):
    """Every requested execution provider failed to load the model."""


class InferenceError(RuntimeError):
    """The session loaded, but one tile did not run."""


def available_providers() -> list[str]:
    return list(ort.get_available_providers())


def cache_is_warm(folder: Path) -> bool:
    """True when the TensorRT cache folder already holds an engine file."""
    try:
        return any(path.is_file() and path.stat().st_size > 0 for path in folder.iterdir())
    except OSError:
        return False


def trt_build_message(folder: Path) -> str:
    """What the first provider attempt should tell the user."""
    if cache_is_warm(folder):
        return "loading the cached TensorRT engine"
    return "building the TensorRT engine"


def open_session(model_path: Path, choices: list[ProviderChoice]) -> tuple[ort.InferenceSession, str]:
    """Load ``model_path`` with the first provider that accepts it."""
    if not choices:
        raise EngineBuildError("no execution provider was requested")
    errors: list[str] = []
    for choice in choices:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=[(choice.name, choice.options)],
            )
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{choice.name}: {exc}")
            continue
        active = list(session.get_providers())
        if choice.name not in active and not (choice.name == CPU and CPU in active):
            errors.append(f"{choice.name}: session used {active}")
            continue
        return session, choice.name
    raise EngineBuildError("; ".join(errors) or "the session did not load")


def _dtype(session: ort.InferenceSession) -> np.dtype[np.generic]:
    kind = str(session.get_inputs()[0].type)
    if "float16" in kind:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def run_tile(session: ort.InferenceSession, tile: np.ndarray) -> np.ndarray:
    """HWC float -> HWC float. The network is NCHW."""
    if tile.ndim != 3:
        raise InferenceError("a tile must be HWC")
    array = np.transpose(tile, (2, 0, 1))[None, ...]
    array = np.ascontiguousarray(array.astype(_dtype(session), copy=False))
    name = str(session.get_inputs()[0].name)
    try:
        result = session.run(None, {name: array})
    except (OSError, RuntimeError, ValueError) as exc:
        raise InferenceError(str(exc)) from exc
    if not result:
        raise InferenceError("the network returned nothing")
    output = np.asarray(result[0])
    if output.ndim == 4:
        output = np.transpose(output[0], (1, 2, 0))
    elif output.ndim != 3:
        raise InferenceError(f"unexpected network output shape {output.shape}")
    return output.astype(np.float32, copy=False)
