"""ONNX Runtime session. One provider at a time, so a failed TensorRT build can fall through."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voxprint_engine.providers import CPU, ProviderChoice
from voxprint_engine.trt_libs import prepare_native_runtime

# CUDA 13 (including nvidia/*/bin/x86_64/cudart64_13.dll) and TensorRT 10
# have to be on the DLL path before this import.
prepare_native_runtime()

import onnxruntime as ort  # noqa: E402

from voxprint_engine.video import _graph_channels, run_batch as _video_run_batch


class EngineBuildError(RuntimeError):
    """Every requested execution provider failed to load the model."""


class InferenceError(RuntimeError):
    """The session loaded, but one tile did not run."""


def available_providers() -> list[str]:
    """ONNX Runtime providers this process can actually load."""
    return list(ort.get_available_providers())


def tensorrt_ep_loaded() -> bool:
    """True when this process can open the TensorRT execution provider."""
    return "TensorrtExecutionProvider" in available_providers()


# Identity network, 1x3x8x8. Small enough that a missing CUDA DLL fails at load
# instead of building an engine. Used to see which provider a session keeps.
_TINY_ONNX = bytes.fromhex(
    "08083a640a190a05696e70757412066f757470757422084964656e74697479120474696e795a1f"
    "0a05696e70757412160a14080112100a0208010a0208030a0208080a02080862200a066f757470"
    "757412160a14080112100a0208010a0208030a0208080a02080842040a00100d"
)
_session_probe: dict[str, tuple[list[str], str]] = {}


def session_providers(requested: str) -> tuple[list[str], str]:
    """Providers a real session kept, and why ``requested`` is absent.

    ``get_available_providers`` reports what the wheel was built with. A missing
    ``cublasLt64_13.dll`` still lists CUDA there, then the session falls back to
    CPU. This opens a tiny graph and reads ``session.get_providers()``.
    """
    cached = _session_probe.get(requested)
    if cached is not None:
        return cached
    try:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(_TINY_ONNX, sess_options=options, providers=[requested])
        active = [str(name) for name in session.get_providers()]
    except (OSError, RuntimeError, ValueError) as exc:
        result: tuple[list[str], str] = ([], str(exc))
        _session_probe[requested] = result
        return result
    reason = ""
    if requested not in active:
        reason = f"{requested} fell back to {active}"
    result = (active, reason)
    _session_probe[requested] = result
    return result


def execution_provider_active(name: str) -> bool:
    """True when a tiny session actually keeps ``name``."""
    active, _reason = session_providers(name)
    return name in active


def cache_is_warm(folder: Path) -> bool:
    """True when this cache already holds a TensorRT ``.engine`` file.

    ORT 1.31 writes ``<cache>/<node>_smXX.engine``. A node name that contains
    a slash nests that file. A ``.profile`` or ``.timing`` file alone is not
    a hit: ORT still builds the engine. One unreadable entry does not hide
    a sibling engine.
    """
    return _engine_file(folder) is not None


def _engine_file(folder: Path, depth: int = 0) -> Path | None:
    if depth > 6:
        return None
    try:
        if not folder.is_dir():
            return None
        children = list(folder.iterdir())
    except OSError:
        return None
    for path in children:
        try:
            if path.is_file() and path.name.lower().endswith(".engine") and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    for path in children:
        try:
            if not path.is_dir():
                continue
        except OSError:
            continue
        found = _engine_file(path, depth + 1)
        if found is not None:
            return found
    return None


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
    expected = _graph_channels(session)
    if expected is not None and int(tile.shape[2]) != expected:
        raise InferenceError(f"this model takes {expected} channels, not {int(tile.shape[2])}")
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


def run_batch(session: ort.InferenceSession, batch: np.ndarray) -> np.ndarray:
    """NCHW float batch. fp16 when the session input type contains float16.

    Image ``run_tile`` is unchanged. This delegates to ``voxprint_engine.video``.
    """
    return _video_run_batch(session, batch)
