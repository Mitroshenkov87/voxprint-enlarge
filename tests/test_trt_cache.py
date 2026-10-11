"""A TensorRT cache hit must say the engine is being loaded, not built."""
from __future__ import annotations

from pathlib import Path

from enlarge.ipc.jobs import _stage_for
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
