"""Progress stages for a TensorRT build and the first-build ETA."""
from __future__ import annotations

from enlarge.ipc.jobs import _advance, _classify, _eta_s, _stage_for
from enlarge.ipc.protocol import STAGES


def test_engine_messages_use_the_new_stages() -> None:
    assert "building_engine" in STAGES
    assert "loading_cached_engine" in STAGES
    assert _stage_for(0.4, "building the TensorRT engine") == "building_engine"
    assert _stage_for(0.4, "loading the cached TensorRT engine") == "loading_cached_engine"
    assert _stage_for(0.45, "TensorRT engine is cached") == "loading_cached_engine"
    assert _stage_for(0.05, "checking the model file") == "loading"
    assert _stage_for(1.0, "ONNX file is cached") == "loading"
    assert _stage_for(0.18, "checked the ONNX file") == "loading"
    assert _stage_for(0.5, "tile 256, overlap 16") == "tiles"
    assert _stage_for(1.0, "onnx TensorrtExecutionProvider") == "encoding"


def test_first_build_eta_is_ninety_seconds_minus_elapsed() -> None:
    assert _eta_s(40.0, "building_engine", started=10.0, now=30.0) == 70.0
    assert _eta_s(40.0, "building_engine", started=0.0, now=100.0) == 0.0
    assert _eta_s(50.0, "tiles", started=0.0, now=10.0) == 10.0
    assert _eta_s(0.5, "loading", started=0.0, now=1.0) is None
    assert _eta_s(100.0, "encoding", started=0.0, now=20.0) == 0.0


def test_stages_do_not_move_backwards() -> None:
    previous = None
    seen: list[str] = []
    for fraction, message in (
        (1.0, "ONNX file is cached"),
        (0.4, "loading the cached TensorRT engine"),
        (0.35, "tile 256, overlap 16"),
        (0.5, "tile 256, overlap 16"),
        (1.0, "onnx CUDAExecutionProvider"),
        (0.2, "converting example.pth to ONNX"),
    ):
        stage = _stage_for(fraction, message)
        nxt = _advance(previous, stage, fraction * 100.0)
        if nxt is None:
            continue
        previous = nxt
        seen.append(nxt[0])
    assert seen == ["loading", "loading_cached_engine", "tiles", "tiles", "encoding"]
    assert _classify("ONNX conversion failed: out of memory") == "model_missing"
    assert _classify("worker closed the pipe: exit -9") == "model_missing"
    assert _classify("out of memory") == "oom"
