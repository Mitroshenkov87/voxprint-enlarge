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

import voxprint_engine.trt_libs as trt_libs
from voxprint_engine.infer import upscale_file
from voxprint_engine.providers import CUDA, TRT
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.session import EngineBuildError, InferenceError
from voxprint_engine.trt_libs import apply_library_path
from voxprint_engine.worker.client import WorkerClient, WorkerError, parse_worker_line
from voxprint_engine.worker.server import _probe_providers, serve


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


def test_native_dirs_keep_cu13_bins_and_tensorrt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "site"
    cublas = root / "nvidia" / "cublas" / "bin"
    cublas.mkdir(parents=True)
    (cublas / "cublasLt64_13.dll").write_bytes(b"x")
    cu12 = root / "nvidia" / "cublas12" / "bin"
    cu12.mkdir(parents=True)
    (cu12 / "cublas64_12.dll").write_bytes(b"x")
    trt = root / "tensorrt_libs"
    trt.mkdir()
    (trt / "nvinfer_10.dll").write_bytes(b"x")
    monkeypatch.setattr(trt_libs, "_site_roots", lambda: [root])
    monkeypatch.setattr(trt_libs, "_distributions", lambda: [])
    monkeypatch.setattr(trt_libs, "_searched", False)
    monkeypatch.setattr(trt_libs, "_folder", None)
    monkeypatch.setattr(trt_libs, "_dirs_ready", False)
    monkeypatch.setattr(trt_libs, "_dirs", [])

    def missing(_name: str) -> object:
        raise trt_libs.metadata.PackageNotFoundError

    monkeypatch.setattr(trt_libs.metadata, "distribution", missing)
    found = [path.resolve() for path in trt_libs.native_library_dirs()]
    assert cublas.resolve() in found
    assert trt.resolve() in found
    assert cu12.resolve() not in found
    assert found.index(cublas.resolve()) < found.index(trt.resolve())


def test_native_dirs_walk_every_cuda13_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "site"
    x86 = root / "nvidia" / "cuda_runtime" / "bin" / "x86_64"
    x86.mkdir(parents=True)
    (x86 / "cudart64_13.dll").write_bytes(b"x")
    cufft = root / "nvidia" / "cufft" / "bin" / "x86_64"
    cufft.mkdir(parents=True)
    (cufft / "cufft64_13.dll").write_bytes(b"x")
    cublas = root / "nvidia" / "cublas" / "bin"
    cublas.mkdir(parents=True)
    (cublas / "cublasLt64_13.dll").write_bytes(b"x")
    cudnn = root / "nvidia" / "cudnn" / "lib"
    cudnn.mkdir(parents=True)
    (cudnn / "libcudnn.so.13").write_bytes(b"x")
    nvjit = root / "nvidia" / "nvjitlink" / "lib" / "x64"
    nvjit.mkdir(parents=True)
    (nvjit / "nvJitLink_13.dll").write_bytes(b"x")
    stale = root / "nvidia" / "cublas12" / "bin"
    stale.mkdir(parents=True)
    (stale / "cublas64_12.dll").write_bytes(b"x")
    empty = root / "nvidia" / "curand" / "bin" / "x86_64"
    empty.mkdir(parents=True)
    (empty / "README.txt").write_bytes(b"x")
    trt = root / "tensorrt_libs"
    trt.mkdir()
    (trt / "nvinfer_10.dll").write_bytes(b"x")
    monkeypatch.setattr(trt_libs, "_site_roots", lambda: [root])
    monkeypatch.setattr(trt_libs, "_distributions", lambda: [])
    monkeypatch.setattr(trt_libs, "_searched", False)
    monkeypatch.setattr(trt_libs, "_folder", None)
    monkeypatch.setattr(trt_libs, "_dirs_ready", False)
    monkeypatch.setattr(trt_libs, "_dirs", [])

    def missing(_name: str) -> object:
        raise trt_libs.metadata.PackageNotFoundError

    monkeypatch.setattr(trt_libs.metadata, "distribution", missing)
    found = [path.resolve() for path in trt_libs.native_library_dirs()]
    expected = [x86, cufft, cublas, cudnn, nvjit, trt]
    assert found == [path.resolve() for path in expected]
    assert stale.resolve() not in found
    assert empty.resolve() not in found


def test_session_import_prepares_before_onnxruntime() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.modules.pop("onnxruntime", None)
import voxprint_engine.trt_libs as trt
seen = {}
real = trt.prepare_native_runtime
def spy():
    seen["loaded"] = "onnxruntime" in sys.modules
    return real()
trt.prepare_native_runtime = spy
import voxprint_engine.session
assert seen.get("loaded") is False
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["VOXPRINT_CI_LANCZOS"] = "1"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_export_and_worker_prepare_before_onnxruntime() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.modules.pop("onnxruntime", None)
import voxprint_engine.trt_libs as trt
seen = []
real = trt.prepare_native_runtime
def spy():
    seen.append("onnxruntime" in sys.modules)
    return real()
trt.prepare_native_runtime = spy
import voxprint_engine.onnx_export as export
try:
    export._load_fp16()
except Exception:
    pass
assert seen and seen[0] is False
for key in list(sys.modules):
    if key == "onnxruntime" or key.startswith("onnxruntime."):
        del sys.modules[key]
seen.clear()
import voxprint_engine.worker.server
assert seen and seen[0] is False
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["VOXPRINT_CI_LANCZOS"] = "1"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_pytest_does_not_reexec_for_linux_libraries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("VOXPRINT_NATIVE_REEXEC", raising=False)
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    assert trt_libs._should_reexec([tmp_path]) is False
    pytest_mod = sys.modules.pop("pytest")
    try:
        assert trt_libs._should_reexec([tmp_path]) is True
    finally:
        sys.modules["pytest"] = pytest_mod
    monkeypatch.setenv("VOXPRINT_NATIVE_REEXEC", "1")
    assert trt_libs._should_reexec([tmp_path]) is False


def test_library_path_prepends_the_tensorrt_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "trt"
    monkeypatch.setattr("voxprint_engine.trt_libs.native_library_dirs", lambda: [folder])
    env: dict[str, str] = {}
    apply_library_path(env)
    key = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    assert env[key].split(os.pathsep)[0] == str(folder)


def test_trt_runtime_failure_uses_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.delenv("VOXPRINT_ORT_DEVICE", raising=False)
    monkeypatch.setattr("voxprint_engine.infer.execution_provider_active", lambda _name: True)
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


def test_inactive_trt_does_not_report_building(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.delenv("VOXPRINT_ORT_DEVICE", raising=False)
    monkeypatch.setattr("voxprint_engine.infer.execution_provider_active", lambda _name: False)
    monkeypatch.setattr("voxprint_engine.infer.available_providers", lambda: [TRT, CUDA])
    notes: list[str] = []

    def open_session(_path: Path, _choices: list[object]) -> tuple[object, str]:
        return object(), CUDA

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
    assert not any("building the TensorRT engine" in text for text in notes)
    assert not any("loading the cached TensorRT engine" in text for text in notes)


def test_cpu_fallback_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.delenv("VOXPRINT_ORT_DEVICE", raising=False)
    monkeypatch.setattr("voxprint_engine.infer.execution_provider_active", lambda _name: False)
    monkeypatch.setattr("voxprint_engine.infer.available_providers", lambda: [TRT, CUDA])
    monkeypatch.setattr("voxprint_engine.infer.torch_available", lambda: False)
    warnings: list[tuple[str, str]] = []

    def open_session(_path: Path, choices: list[object]) -> tuple[object, str]:
        name = str(getattr(choices[0], "name", ""))
        raise EngineBuildError(f"{name}: session used ['CPUExecutionProvider']")

    monkeypatch.setattr("voxprint_engine.infer.open_session", open_session)
    src = tmp_path / "in.png"
    Image.new("RGB", (8, 8), (4, 5, 6)).save(src)
    (tmp_path / "net.onnx").write_bytes(b"onnx")
    with pytest.raises(Exception, match="PyTorch"):
        upscale_file(
            src,
            tmp_path / "out.avif",
            _manifest(),
            scale=2,
            image_format="avif",
            quality=60,
            double=False,
            device="auto",
            user_tile=32,
            batch=1,
            onnx_path=tmp_path / "net.onnx",
            on_warning=lambda code, text: warnings.append((code, text)),
        )
    assert warnings
    assert warnings[0][0] == "gpu_provider_unavailable"
    assert "CPUExecutionProvider" in warnings[0][1]


def test_worker_probe_reports_the_session_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(name: str) -> tuple[list[str], str]:
        if name == CUDA:
            return [CUDA, "CPUExecutionProvider"], ""
        return ["CPUExecutionProvider"], f"{name} fell back to ['CPUExecutionProvider']"

    monkeypatch.setattr("voxprint_engine.worker.server.session_providers", fake)
    result = _probe_providers({"id": "selftest", "op": "probe_providers"})
    assert result["ok"] is False
    rows = result["providers"]
    assert isinstance(rows, list)
    by_name = {row["name"]: row for row in rows}
    assert by_name[CUDA]["ok"] is True
    assert by_name[TRT]["providers"] == ["CPUExecutionProvider"]
    assert "fell back" in by_name[TRT]["reason"]
