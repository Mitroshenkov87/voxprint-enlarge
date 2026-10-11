"""A TensorRT cache hit must say the engine is being loaded, not built."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from enlarge.ipc.jobs import _stage_for
from voxprint_engine.infer import upscale_file
from voxprint_engine.providers import CUDA, TRT
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.session import cache_is_warm, trt_build_message


def test_a_nested_engine_file_is_a_cache_hit(tmp_path: Path) -> None:
    nested = tmp_path / "node" / "subgraph"
    nested.mkdir(parents=True)
    (nested / "Conv_sm89.engine").write_bytes(b"engine")
    (tmp_path / "profile.profile").write_text("cold", encoding="utf-8")
    assert cache_is_warm(tmp_path) is True
    assert trt_build_message(tmp_path) == "loading the cached TensorRT engine"
    assert _stage_for(0.4, trt_build_message(tmp_path)) == "loading_cached_engine"


def test_profile_and_timing_files_are_not_a_cache_hit(tmp_path: Path) -> None:
    (tmp_path / "node.profile").write_text("profile", encoding="utf-8")
    (tmp_path / "node.timing").write_text("timing", encoding="utf-8")
    assert cache_is_warm(tmp_path) is False
    assert trt_build_message(tmp_path) == "building the TensorRT engine"
    assert _stage_for(0.4, trt_build_message(tmp_path)) == "building_engine"


def test_an_empty_engine_file_is_not_a_hit(tmp_path: Path) -> None:
    (tmp_path / "empty.engine").write_bytes(b"")
    assert cache_is_warm(tmp_path) is False


def _manifest() -> ModelManifest:
    return ModelManifest(
        id="4x-example-model",
        name="Example",
        arch="rrdb",
        scale=2,
        role="upscale",
        licence="MIT",
        author="Someone",
        attribution="Example by Someone, MIT.",
        source_url="https://example.com/model",
        sha256="a" * 64,
        vram_mb=1000,
        tile=32,
        filename="example.pth",
        summary="A test manifest.",
        selectable=True,
        download_url="https://example.com/example.pth",
    )


def test_a_warm_cache_reports_loading_when_tensorrt_stays_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.delenv("VOXPRINT_ORT_DEVICE", raising=False)
    cache = tmp_path / "engines"
    cache.mkdir()
    (cache / "Conv_sm89.engine").write_bytes(b"engine")
    monkeypatch.setattr("voxprint_engine.providers.trt_cache_dir", lambda: cache)
    monkeypatch.setattr("voxprint_engine.infer.execution_provider_active", lambda name: name == TRT)
    monkeypatch.setattr("voxprint_engine.infer.available_providers", lambda: [TRT, CUDA])
    notes: list[str] = []

    def open_session(_path: Path, choices: list[object]) -> tuple[object, str]:
        assert getattr(choices[0], "name", "") == TRT
        return object(), TRT

    def run_tile(_session: object, tile: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

    monkeypatch.setattr("voxprint_engine.infer.open_session", open_session)
    monkeypatch.setattr("voxprint_engine.infer.run_tile", run_tile)
    src = tmp_path / "in.png"
    dst = tmp_path / "out.avif"
    Image.new("RGB", (8, 8), (4, 5, 6)).save(src)
    (tmp_path / "net.onnx").write_bytes(b"onnx")
    upscale_file(
        src,
        dst,
        _manifest(),
        scale=2,
        image_format="avif",
        quality=60,
        double=False,
        device="auto",
        user_tile=32,
        batch=1,
        onnx_path=tmp_path / "net.onnx",
        on_progress=lambda _fraction, text: notes.append(text),
    )
    assert "loading the cached TensorRT engine" in notes
    assert "building the TensorRT engine" not in notes
    assert _stage_for(0.4, "loading the cached TensorRT engine") == "loading_cached_engine"
