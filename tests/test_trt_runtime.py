"""TensorRT 10 pin, protocol noise, and the CUDA fallback after a runtime EP failure."""
from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from voxprint_engine.infer import upscale_file
from voxprint_engine.providers import CUDA, TRT
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.session import InferenceError
from voxprint_engine.trt_libs import apply_library_path
from voxprint_engine.worker.client import WorkerClient, WorkerError, parse_worker_line
from voxprint_engine.worker.server import serve


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


def test_ep_error_banner_is_not_protocol() -> None:
    assert parse_worker_line("*************** EP Error ***************\n") is None
    assert parse_worker_line("EP Error: [ONNXRuntimeError] : 11 : EP_FAIL : TRTKernel\n") is None
    ready = parse_worker_line('{"op": "ready", "engine": "voxprint"}\n')
    assert ready is not None and ready["op"] == "ready"
    with pytest.raises(WorkerError, match="malformed json"):
        parse_worker_line("{not-json")


def test_client_read_skips_injected_garbage() -> None:
    client = WorkerClient()
    client._lines.put("*************** EP Error ***************\n")
    client._lines.put("EP Error: [ONNXRuntimeError] : 11 : EP_FAIL : TRTKernel_graph_main_graph\n")
    client._lines.put('{"op": "ready", "engine": "voxprint"}\n')
    message = client._read(1)
    assert message["op"] == "ready"


def test_injected_stream_does_not_redirect_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_dup(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("a provided stdout must stay on that stream")

    monkeypatch.setattr(os, "dup", refuse_dup)
    code = serve(stdin=io.StringIO('{"op": "shutdown"}\n'), stdout=io.StringIO())
    assert code == 0


def test_native_prints_stay_off_the_protocol_pipe(tmp_path: Path) -> None:
    script = """
import os, sys
os.environ["VOXPRINT_HOME"] = sys.argv[1]
from voxprint_engine.worker.server import isolate_native_stdio
protocol = isolate_native_stdio()
protocol.write('{"op":"ready"}\\n')
protocol.flush()
print("*************** EP Error ***************", flush=True)
sys.stderr.write("EP Error: stderr\\n")
sys.stderr.flush()
"""
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == '{"op":"ready"}'
    assert "EP Error" not in proc.stdout
    log = (tmp_path / "logs" / "worker-native.log").read_text(encoding="utf-8")
    assert "EP Error" in log
    assert "stderr" in log


def test_library_path_prepends_the_tensorrt_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "trt"
    monkeypatch.setattr("voxprint_engine.trt_libs.library_dir", lambda: folder)
    env: dict[str, str] = {}
    apply_library_path(env)
    key = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    assert env[key].split(os.pathsep)[0] == str(folder)


def test_trt_runtime_failure_uses_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.delenv("VOXPRINT_ORT_DEVICE", raising=False)
    monkeypatch.setattr("voxprint_engine.infer.available_providers", lambda: [TRT, CUDA])
    opened: list[str] = []

    def open_session(_path: Path, choices: list[object]) -> tuple[object, str]:
        name = str(getattr(choices[0], "name", ""))
        opened.append(name)
        return object(), name

    calls = {"n": 0}

    def run_tile(_session: object, tile: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] == 1:
            raise InferenceError("EP_FAIL : TRTKernel_graph_main_graph")
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

    monkeypatch.setattr("voxprint_engine.infer.open_session", open_session)
    monkeypatch.setattr("voxprint_engine.infer.run_tile", run_tile)
    src = tmp_path / "in.png"
    dst = tmp_path / "out.avif"
    Image.new("RGB", (8, 8), (4, 5, 6)).save(src)
    (tmp_path / "net.onnx").write_bytes(b"onnx")
    outcome = upscale_file(
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
    )
    assert opened == [TRT, CUDA]
    assert CUDA in outcome.message
    assert dst.is_file()
