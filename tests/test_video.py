"""Video helpers. No GPU: the session is a double, and the worker handler is patched."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from voxprint_engine import API
from voxprint_engine.gpu_lock import gpu_job, lock_path, read_lock, release, try_acquire
from voxprint_engine.hardware import OLD_DRIVER_MESSAGE, assess_gpu
from voxprint_engine.providers import CPU, CUDA, TRT
from voxprint_engine.session import EngineBuildError, run_batch as session_run_batch
from voxprint_engine.tile import upscale_tiled
from voxprint_engine.video import (
    SPILL_BYTES,
    RestoreBatchRequest as VideoRestoreBatchRequest,
    gather_window,
    parse_shared_usage,
    read_shared_usage,
    restore_frames,
    run_batch,
    sysmem_spill,
    temporal_windows,
    tile_frames,
    to_nchw,
    to_nhwc,
)
from voxprint_engine.vram import sysmem_spill as vram_sysmem_spill
from voxprint_engine.worker.protocol import RestoreBatchRequest
from voxprint_engine.worker.server import handle


def _scale_hwc(scale: int):
    def infer(tile: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)

    return infer


def _scale_nchw(scale: int):
    def infer(batch: np.ndarray) -> np.ndarray:
        frames = np.transpose(batch, (0, 2, 3, 1))
        scaled = np.stack([_scale_hwc(scale)(frame) for frame in frames], axis=0)
        return np.ascontiguousarray(np.transpose(scaled, (0, 3, 1, 2)))

    return infer


def _kept_sources(frame_count: int, length: int, overlap: int) -> list[int]:
    sources: list[int] = []
    for window in temporal_windows(frame_count, length, overlap):
        for slot in range(window.keep_start, window.keep_stop):
            sources.append(window.input_start + slot)
    return sources


def _smi(stdout: str, code: int = 0):
    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=code, stdout=stdout, stderr="")

    return runner


class _Input:
    def __init__(self, name: str, type_name: str) -> None:
        self.name = name
        self.type = type_name


class _Session:
    def __init__(self, type_name: str = "tensor(float)", output: np.ndarray | None = None, error: BaseException | None = None) -> None:
        self.type_name = type_name
        self.output = output
        self.error = error
        self.seen: list[np.ndarray] = []

    def get_inputs(self) -> list[_Input]:
        return [_Input("input", self.type_name)]

    def run(self, _names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        array = feeds["input"]
        self.seen.append(array)
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return [self.output]
        return [array.astype(np.float32)]


def test_api_stays_at_1() -> None:
    assert API == 1
    assert VideoRestoreBatchRequest is RestoreBatchRequest


def test_framewise_and_temporal_windows_emit_each_frame_once() -> None:
    assert _kept_sources(4, 1, 0) == [0, 1, 2, 3]
    assert len(temporal_windows(4, 1, 0)) == 4
    for count, length, overlap in ((1, 1, 0), (5, 3, 1), (2, 4, 2), (4, 3, 2), (7, 4, 1)):
        assert _kept_sources(count, length, overlap) == list(range(count))
    with pytest.raises(ValueError, match="frame_count"):
        temporal_windows(0, 1, 0)
    with pytest.raises(ValueError, match="length"):
        temporal_windows(2, 0, 0)
    with pytest.raises(ValueError, match="overlap"):
        temporal_windows(4, 3, 3)


def test_gather_window_pads_with_the_edge_frame() -> None:
    frames = np.arange(3, dtype=np.float32)[:, None]
    window = temporal_windows(3, 3, 1)[0]
    assert window.input_start == -1
    assert window.pad_left == 1
    gathered = gather_window(frames, window)
    assert gathered.shape == (3, 1)
    assert np.allclose(gathered[:, 0], [0, 0, 1])


def test_nchw_batch_casts_fp16_and_returns_float32() -> None:
    image = np.zeros((2, 4, 5, 3), dtype=np.float32)
    image[0, 0, 0] = (0.25, 0.5, 0.75)
    batch = to_nchw(image)
    assert batch.shape == (2, 3, 4, 5)
    assert np.allclose(to_nhwc(batch), image)
    session = _Session("tensor(float16)")
    output = run_batch(session, batch)
    assert output.dtype == np.float32
    assert output.shape == batch.shape
    assert session.seen[0].dtype == np.float16
    assert np.allclose(output[0, :, 0, 0], [0.25, 0.5, 0.75])
    assert np.allclose(session_run_batch(session, batch), run_batch(session, batch))


def test_run_batch_reports_a_failed_session() -> None:
    session = _Session(error=RuntimeError("failed to allocate"))
    with pytest.raises(RuntimeError, match="failed to allocate"):
        run_batch(session, np.zeros((1, 1, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="NCHW"):
        run_batch(session, np.zeros((2, 2), dtype=np.float32))
    empty = _Session(output=np.zeros((1, 1, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="unexpected network output"):
        run_batch(empty, np.zeros((1, 1, 2, 2), dtype=np.float32))


def test_tile_frames_match_the_image_feather_and_batch_on_n() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((20, 30, 3), dtype=np.float32)
    tiled = tile_frames(image[None, ...], _scale_nchw(1), tile=16, overlap=4, scale=1, batch=3)[0]
    assert np.allclose(tiled, upscale_tiled(image, _scale_hwc(1), 16, 4, 1), atol=1e-5)
    assert np.allclose(tiled, image, atol=1e-5)
    flat = np.full((8, 10, 3), 0.5, dtype=np.float32)
    scaled = tile_frames(flat[None, ...], _scale_nchw(2), tile=8, overlap=0, scale=2, batch=1)[0]
    assert scaled.shape == (16, 20, 3)
    assert np.allclose(scaled, 0.5)

    seen: list[int] = []

    def infer(batch: np.ndarray) -> np.ndarray:
        seen.append(int(batch.shape[0]))
        assert batch.shape[1:] == (3, 16, 16)
        return batch

    frames = rng.random((2, 20, 30, 3), dtype=np.float32)
    out = tile_frames(frames, infer, tile=16, overlap=4, scale=1, batch=4)
    assert seen == [4, 4, 4]
    assert out.shape == frames.shape
    once = tile_frames(frames, infer, tile=16, overlap=4, scale=1, batch=1)
    assert np.allclose(out, once, atol=1e-5)


def test_restore_frames_keeps_each_source_frame() -> None:
    rng = np.random.default_rng(1)
    frames = rng.random((5, 18, 22, 3), dtype=np.float32)
    restored = restore_frames(
        frames,
        _scale_nchw(1),
        tile=16,
        overlap=4,
        scale=1,
        batch=3,
        length=3,
        window_overlap=1,
    )
    assert restored.shape == frames.shape
    assert np.allclose(restored, frames, atol=1e-5)
    framewise = restore_frames(frames, _scale_nchw(1), tile=16, overlap=4, scale=1, batch=2, length=1, window_overlap=0)
    assert np.allclose(framewise, frames, atol=1e-5)


def test_sysmem_spill_is_silent_without_both_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    assert vram_sysmem_spill(None, 0) is False
    assert vram_sysmem_spill(10, None) is False
    assert vram_sysmem_spill(SPILL_BYTES, 0) is False
    assert vram_sysmem_spill(SPILL_BYTES + 1, 0) is True
    assert sysmem_spill(5, 0, threshold=5) is False
    text = '"stamp","10"\n"(PDH-CSV 4.0)","\\\\GPU Process Memory(pid_1*)\\\\Shared Usage"\n"10/10/2026","4096"\n'
    assert parse_shared_usage(text) == 4096
    assert parse_shared_usage("") is None
    monkeypatch.setattr("voxprint_engine.video.os.name", "posix")
    assert read_shared_usage() is None


def test_read_shared_usage_parses_typeperf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voxprint_engine.video.os.name", "nt")

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[0] == "typeperf"
        assert "pid_7" in command[1]
        return subprocess.CompletedProcess(command, 0, stdout='"t","100"\n"t","4096"\n', stderr="")

    monkeypatch.setattr("voxprint_engine.video.subprocess.run", runner)
    assert read_shared_usage(pid=7) == 4096

    def missing(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("no typeperf")

    monkeypatch.setattr("voxprint_engine.video.subprocess.run", missing)
    assert read_shared_usage(pid=7) is None


def test_refuse_below_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    quiet = assess_gpu(runner=_smi("NVIDIA GeForce RTX 4090, 610.73.01, N/A\n"))
    assert quiet.ok is True
    assert quiet.message == ""
    missing = assess_gpu(runner=_smi("NVIDIA GeForce RTX 4090, 610.73.01, N/A\n"), refuse_below=(8, 9))
    assert missing.ok is False
    assert "did not report" in missing.message
    assert missing.warning == ""
    refused = assess_gpu(runner=_smi("NVIDIA GeForce RTX 3090, 610.73.01, 8.6\n"), refuse_below=(8, 9))
    assert refused.ok is False
    assert "8.6" in refused.message
    assert "8.9" in refused.message
    warned = assess_gpu(runner=_smi("NVIDIA GeForce RTX 3090, 610.73.01, 8.6\n"))
    assert warned.ok is True
    assert "8.6" in warned.warning
    allowed = assess_gpu(runner=_smi("NVIDIA GeForce RTX 4090, 610.73.01, 8.9\n"), refuse_below=(8, 9))
    assert allowed.ok is True
    assert allowed.warning == ""
    old = assess_gpu(runner=_smi("NVIDIA GeForce RTX 4090, 580.88, 8.9\n"), refuse_below=(8, 9))
    assert old.ok is False
    assert old.message == OLD_DRIVER_MESSAGE.format(driver="580.88")


def test_ci_flag_skips_refuse_below(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_CI_LANCZOS", "1")

    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("nvidia-smi must not run")

    gate = assess_gpu(runner=runner, refuse_below=(8, 9))
    assert gate.ok is True
    assert gate.message == ""


def test_gpu_lock_owner_defaults_to_enlarge() -> None:
    with gpu_job("preview"):
        held = read_lock()
        assert held is not None
        assert held["owner"] == "enlarge"
    assert read_lock() is None

    assert try_acquire("clip", owner="clarity") is None
    held = read_lock()
    assert held is not None and held["owner"] == "clarity"
    blocked = try_acquire("photo")
    assert blocked is not None and blocked["owner"] == "clarity"
    release()
    assert read_lock() is not None
    with pytest.raises(TimeoutError):
        with gpu_job("photo", timeout=0.05, poll=0.01):
            pass
    release(owner="clarity")
    assert read_lock() is None

    with gpu_job("clip", owner="clarity"):
        with gpu_job("inner"):
            nested = read_lock()
            assert nested is not None
            assert nested["owner"] == "clarity"
            assert nested["job"] == "clip"
    assert read_lock() is None


def test_a_child_shares_a_parent_lock_only_for_the_same_owner() -> None:
    target = lock_path()
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    target.write_text(
        json.dumps(
            {
                "owner": "clarity",
                "pid": os.getppid(),
                "job": "clarity:restore",
                "started": future,
                "eta": future,
            }
        ),
        encoding="utf-8",
    )
    assert try_acquire("child", owner="clarity") is None
    release(owner="clarity")
    held = read_lock()
    assert held is not None and held.get("pid") == os.getppid()
    blocked = try_acquire("photo")
    assert blocked is not None and blocked["owner"] == "clarity"
    release()
    assert read_lock() is not None


def test_restore_batch_uses_the_session_lock_and_trt_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_TRT_CACHE", str(tmp_path / "trt"))
    frames = np.linspace(0.0, 1.0, 2 * 8 * 8 * 3, dtype=np.float32).reshape(2, 8, 8, 3)
    src = tmp_path / "frames.npy"
    dst = tmp_path / "out" / "restored.npy"
    np.save(src, frames)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    opened: dict[str, object] = {}

    class _LockedSession:
        def get_inputs(self) -> list[_Input]:
            return [_Input("input", "tensor(float)")]

        def run(self, _names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            lock = read_lock()
            assert lock is not None
            assert lock["owner"] == "clarity"
            assert lock["job"] == "clarity:restore"
            assert feeds["input"].shape[0] == 2
            return [feeds["input"]]

        def close(self) -> None:
            opened["closed"] = True

    def opener(path: Path, choices: list[object]) -> tuple[_LockedSession, str]:
        choice = choices[0]
        opened["path"] = path
        opened["provider"] = choice.name
        opened["fp16"] = choice.options.get("trt_fp16_enable")
        opened["cache"] = choice.options.get("trt_engine_cache_path")
        return _LockedSession(), str(choice.name)

    monkeypatch.setattr("voxprint_engine.worker.restore.available_providers", lambda: [TRT, CUDA, CPU])
    monkeypatch.setattr("voxprint_engine.worker.restore.open_session", opener)
    reply = handle(
        {
            "op": "restore_batch",
            "id": "clip-1",
            "frames_npy": str(src),
            "out_npy": str(dst),
            "onnx": str(onnx),
            "tile": 16,
            "overlap": 4,
            "batch": 2,
            "scale": 1,
            "length": 1,
            "window_overlap": 0,
            "device": "auto",
            "owner": "clarity",
        }
    )
    assert reply is not None
    assert reply["op"] == "result"
    assert reply["ok"] is True
    assert reply["id"] == "clip-1"
    assert reply["engine"] == "onnx"
    assert reply["frames"] == 2
    assert reply["dst"] == str(dst)
    assert TRT in str(reply["message"])
    assert opened["provider"] == TRT
    assert opened["fp16"] == "True"
    assert opened["path"] == onnx
    cache = Path(str(opened["cache"]))
    assert cache.is_dir()
    assert opened["closed"] is True
    assert read_lock() is None
    assert np.allclose(np.load(dst), frames, atol=1e-5)

    request = RestoreBatchRequest(id="clip-1", frames_npy=str(src), out_npy=str(dst), onnx=str(onnx))
    message = request.to_message()
    assert message["op"] == "restore_batch"
    assert message["owner"] == "clarity"
    assert message["length"] == 1
    assert message["tile"] == 0


def test_restore_batch_defaults_a_missing_tile_and_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_TRT_CACHE", str(tmp_path / "trt"))
    frames = np.zeros((1, 4, 4, 3), dtype=np.float32)
    src = tmp_path / "frames.npy"
    dst = tmp_path / "out.npy"
    np.save(src, frames)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    seen: dict[str, object] = {}

    class _DefaultSession:
        def get_inputs(self) -> list[_Input]:
            return [_Input("input", "tensor(float)")]

        def run(self, _names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            seen["shape"] = feeds["input"].shape
            lock = read_lock()
            assert lock is not None
            seen["owner"] = lock["owner"]
            return [feeds["input"]]

        def close(self) -> None:
            return None

    monkeypatch.setattr("voxprint_engine.worker.restore.available_providers", lambda: [TRT, CUDA, CPU])
    monkeypatch.setattr(
        "voxprint_engine.worker.restore.open_session",
        lambda _path, choices: (_DefaultSession(), str(choices[0].name)),
    )
    reply = handle(
        {
            "op": "restore_batch",
            "id": "defaults",
            "frames_npy": str(src),
            "out_npy": str(dst),
            "onnx": str(onnx),
            "device": "auto",
        }
    )
    assert reply is not None and reply["ok"] is True
    assert seen["shape"] == (1, 3, 512, 512)
    assert seen["owner"] == "clarity"
    assert read_lock() is None


def test_restore_batch_uses_cuda_when_tensorrt_does_not_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_TRT_CACHE", str(tmp_path / "trt"))
    frames = np.zeros((1, 4, 4, 3), dtype=np.float32)
    src = tmp_path / "frames.npy"
    dst = tmp_path / "out.npy"
    np.save(src, frames)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    tried: list[str] = []

    class _CudaSession:
        def get_inputs(self) -> list[_Input]:
            return [_Input("input", "tensor(float)")]

        def run(self, _names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            return [feeds["input"]]

        def close(self) -> None:
            tried.append("closed")

    def opener(_path: Path, choices: list[object]) -> tuple[_CudaSession, str]:
        lock = read_lock()
        assert lock is not None and lock["owner"] == "clarity"
        name = str(choices[0].name)
        tried.append(name)
        if name == TRT:
            raise EngineBuildError("trt engine failed")
        return _CudaSession(), name

    monkeypatch.setattr("voxprint_engine.worker.restore.available_providers", lambda: [TRT, CUDA, CPU])
    monkeypatch.setattr("voxprint_engine.worker.restore.open_session", opener)
    reply = handle(
        {
            "op": "restore_batch",
            "id": "fallback",
            "frames_npy": str(src),
            "out_npy": str(dst),
            "onnx": str(onnx),
            "tile": 8,
            "batch": 1,
            "scale": 1,
            "device": "auto",
            "owner": "clarity",
        }
    )
    assert reply is not None and reply["ok"] is True
    assert tried[:2] == [TRT, CUDA]
    assert tried[-1] == "closed"
    assert CUDA in str(reply["message"])
    assert read_lock() is None

    def fail(_path: Path, choices: list[object]) -> tuple[_CudaSession, str]:
        raise EngineBuildError(str(choices[0].name))

    monkeypatch.setattr("voxprint_engine.worker.restore.open_session", fail)
    failed = handle(
        {
            "op": "restore_batch",
            "id": "both",
            "frames_npy": str(src),
            "out_npy": str(dst),
            "onnx": str(onnx),
            "tile": 8,
            "device": "auto",
        }
    )
    assert failed is not None and failed["op"] == "error"
    assert TRT in failed["message"]
    assert CUDA in failed["message"]
    assert read_lock() is None


def test_restore_batch_reports_errors_and_leaves_upscale_dispatch(tmp_path: Path) -> None:
    missing = handle(
        {"op": "restore_batch", "id": "missing", "frames_npy": str(tmp_path / "no.npy"), "onnx": str(tmp_path / "no.onnx")}
    )
    assert missing is not None
    assert missing["op"] == "error"
    assert missing["ok"] is False
    assert missing["id"] == "missing"
    src = tmp_path / "flat.npy"
    np.save(src, np.zeros((2, 2), dtype=np.float32))
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    flat = handle({"op": "restore_batch", "id": "flat", "frames_npy": str(src), "out_npy": str(tmp_path / "o.npy"), "onnx": str(onnx)})
    assert flat is not None
    assert flat["op"] == "error"
    assert "NHWC" in flat["message"]
    unknown = handle({"op": "nope", "id": "z"})
    assert unknown is not None
    assert unknown["message"] == "unknown op 'nope'"
    ping = handle({"op": "ping", "id": "p"})
    assert ping is not None
    assert ping["message"] == "pong"
