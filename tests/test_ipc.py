"""Protocol 1: capabilities, discovery, socket jobs, permissions, cancel, and names."""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PIL import Image
from PySide6.QtWidgets import QApplication

from enlarge.ipc.naming import pipe_name, reserve_default, socket_path
from enlarge.ipc.protocol import dumps, is_retryable, loads
from enlarge.ipc.register import write_linux
from enlarge.ipc.server import IpcServer
from main import main
from voxprint_engine.formats import heic_backend
from voxprint_engine.hardware import GpuInfo


def _app() -> QApplication:
    running = QApplication.instance()
    if isinstance(running, QApplication):
        return running
    return QApplication([])


def _picture(path: Path) -> None:
    Image.new("RGB", (8, 6), (12, 40, 90)).save(path)


def _pump(thread: threading.Thread, timeout: float = 20.0) -> None:
    app = _app()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1)
    assert not thread.is_alive()


def _read_until(sock: socket.socket, ops: set[str], timeout: float = 20.0) -> list[dict[str, object]]:
    sock.settimeout(0.2)
    buffer = b""
    messages: list[dict[str, object]] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(65536)
        except TimeoutError:
            chunk = b""
        if not chunk:
            continue
        buffer += chunk
        while b"\n" in buffer:
            raw, _, buffer = buffer.partition(b"\n")
            message = loads(raw)
            messages.append(message)
            if message.get("op") in ops:
                return messages
    raise AssertionError(messages)


def test_pipe_name_uses_the_sid_hash() -> None:
    name = pipe_name("S-1-5-21-1-2-3-1001")
    assert name.startswith("voxprint-enlarge-")
    assert len(name.removeprefix("voxprint-enlarge-")) == 16
    assert name == pipe_name("S-1-5-21-1-2-3-1001")
    assert name != pipe_name("S-1-5-21-1-2-3-1002")


def test_retryable_flags() -> None:
    assert is_retryable("gpu_busy") and is_retryable("model_missing")
    assert not is_retryable("oom")
    assert not is_retryable("io_error")
    assert not is_retryable("internal")
    assert not is_retryable("gpu_unsupported")
    assert not is_retryable("cancelled")
    assert not is_retryable("bad_input")
    assert not is_retryable("unsupported_format")


def test_default_names_never_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    first = reserve_default(source, 2, ".avif")
    second = reserve_default(source, 2, ".avif")
    assert first.name == "hill.enlarged-2x.avif"
    assert second.name == "hill.enlarged-2x-2.avif"
    assert first.read_bytes() == b""
    assert second.is_file()


def test_capabilities_with_no_gpu(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("enlarge.ipc.capabilities.query_gpus", lambda runner=None: [])
    assert main(["--capabilities", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == 1
    assert payload["app"] == "enlarge"
    assert payload["presets"] == ["photo", "art"]
    assert payload["max_scale"] == 4
    assert payload["gpu"] == {"name": None, "cc": None, "status": "none", "reason": "no_nvidia"}
    assert payload["default_output_format"] == "avif"
    assert payload["voxprint_home_source"] == "env"
    assert payload["voxprint_home"].endswith("voxprint")
    assert "png" not in payload["output_formats"]
    expected = ["avif", "jxl", "webp", "jpeg"]
    if heic_backend() is not None:
        expected.append("heic")
    assert payload["output_formats"] == expected
    assert "png" in payload["input_formats"]
    presets = {row["preset"] for row in payload["models"]}
    assert presets == {"photo", "art"}
    for row in payload["models"]:
        assert row["quality"] in {"fast", "max"}
        assert row["scales"] == [2, 4]
        assert row["installed"] is False
        assert row["id"]


def test_capabilities_report_an_old_driver_and_a_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "enlarge.ipc.capabilities.query_gpus",
        lambda runner=None: [GpuInfo("NVIDIA GeForce RTX 4090", "560.35.03", (8, 9))],
    )
    assert main(["--capabilities", "--json"]) == 0
    old = json.loads(capsys.readouterr().out)
    assert old["gpu"] == {
        "name": "NVIDIA GeForce RTX 4090",
        "cc": "8.9",
        "status": "none",
        "reason": "driver_too_old",
    }
    monkeypatch.setattr(
        "enlarge.ipc.capabilities.query_gpus",
        lambda runner=None: [GpuInfo("NVIDIA GeForce RTX 3090", "610.73.01", (8, 6))],
    )
    assert main(["--capabilities", "--json"]) == 0
    warn = json.loads(capsys.readouterr().out)
    assert warn["gpu"] == {
        "name": "NVIDIA GeForce RTX 3090",
        "cc": "8.6",
        "status": "warn",
        "reason": None,
    }
    monkeypatch.setattr(
        "enlarge.ipc.capabilities.query_gpus",
        lambda runner=None: [GpuInfo("NVIDIA GeForce RTX 4090", "610.73.01", (8, 9))],
    )
    assert main(["--capabilities", "--json"]) == 0
    ok = json.loads(capsys.readouterr().out)
    assert ok["gpu"] == {
        "name": "NVIDIA GeForce RTX 4090",
        "cc": "8.9",
        "status": "ok",
        "reason": None,
    }


def test_linux_discovery_files(tmp_path: Path) -> None:
    exe = tmp_path / "main.py"
    exe.write_text("print('enlarge')\n", encoding="utf-8")
    desktop = write_linux(exe, tmp_path / "share", tmp_path / "bin", ipc_name="/run/user/1/voxprint/enlarge.sock")
    text = desktop.read_text(encoding="utf-8")
    assert desktop.name == "ai.voxprint.Enlarge.desktop"
    assert "X-Voxprint-Ipc=/run/user/1/voxprint/enlarge.sock" in text
    assert "X-Voxprint-Protocol=1" in text
    assert "X-Voxprint-Ipc-Base=voxprint-enlarge" in text
    assert "X-Voxprint-Version=0.2.0-rc.5" in text
    assert f"X-Voxprint-Exe={exe}" in text
    launcher = tmp_path / "bin" / "voxprint-enlarge"
    assert os.access(launcher, os.X_OK)
    assert "main.py" in launcher.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="the fake client connects to the Linux socket; Windows listens on a named pipe")
def test_socket_job_permissions_and_unknown_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    source = tmp_path / "hill.png"
    _picture(source)
    taken = tmp_path / "hill.enlarged-2x.avif"
    taken.write_bytes(b"keep")
    _app()
    server = IpcServer()
    assert server.listen()
    endpoint = Path(server.endpoint())
    assert endpoint == socket_path()
    assert endpoint.parent.stat().st_mode & 0o777 == 0o700
    assert endpoint.stat().st_mode & 0o777 == 0o600
    holder: dict[str, object] = {}

    def client() -> None:
        sock = socket.socket(socket.AF_UNIX)
        sock.connect(str(endpoint))
        sock.sendall(dumps({"op": "hello", "client": "lucent", "future": True}))
        hello = _read_until(sock, {"hello"})
        sock.sendall(
            dumps(
                {
                    "op": "enlarge.request",
                    "id": "job-1",
                    "input": str(source),
                    "scale": 2,
                    "preset": "photo",
                    "quality": "fast",
                    "format": "avif",
                    "open_ui": True,
                    "note": "ignored",
                }
            )
        )
        rest = _read_until(sock, {"done", "error"})
        sock.close()
        holder["messages"] = hello + rest

    thread = threading.Thread(target=client)
    thread.start()
    _pump(thread)
    server.close()
    if server.window is not None:
        server.window.close()
    messages = holder["messages"]
    assert isinstance(messages, list)
    ops = [message["op"] for message in messages]
    assert ops[0] == "hello"
    assert messages[0]["protocol_min"] == 1 and messages[0]["protocol_max"] == 1
    assert "accepted" in ops
    assert "done" in ops
    done = next(message for message in messages if message["op"] == "done")
    output = Path(str(done["output"]))
    assert output.name == "hill.enlarged-2x-2.avif"
    assert output.is_file() and output.stat().st_size > 0
    assert taken.read_bytes() == b"keep"
    assert done["width"] == 16 and done["height"] == 12
    assert done["format"] == "avif"
    assert all(message["protocol"] == 1 for message in messages)
    stages = [message["stage"] for message in messages if message["op"] == "progress"]
    assert "queued" in stages
    assert "encoding" in stages
    assert server.window is not None
    assert any(path == source for path in server.window.sources)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake client connects to the Linux socket; Windows listens on a named pipe")
def test_cancel_stops_the_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("VOXPRINT_FAKE_DELAY", "1.5")
    source = tmp_path / "slow.png"
    _picture(source)
    _app()
    server = IpcServer()
    assert server.listen()
    endpoint = server.endpoint()
    holder: dict[str, object] = {}

    def client() -> None:
        sock = socket.socket(socket.AF_UNIX)
        sock.connect(endpoint)
        sock.sendall(
            dumps(
                {
                    "op": "enlarge.request",
                    "id": "slow-1",
                    "input": str(source),
                    "scale": 4,
                    "preset": "art",
                    "quality": "max",
                }
            )
        )
        early = _read_until(sock, {"accepted", "error"})
        assert early[-1]["op"] == "accepted"
        sock.sendall(dumps({"op": "cancel", "id": "slow-1"}))
        rest = _read_until(sock, {"done", "error"})
        sock.close()
        holder["messages"] = early + rest

    started = time.monotonic()
    thread = threading.Thread(target=client)
    thread.start()
    _pump(thread)
    server.close()
    messages = holder["messages"]
    assert isinstance(messages, list)
    error = next(message for message in messages if message["op"] == "error")
    assert error["code"] == "cancelled"
    assert error["retryable"] is False
    assert time.monotonic() - started < 8
    assert not (tmp_path / "slow.enlarged-4x.avif").exists()


def test_cli_json_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "in.png"
    _picture(source)
    code = main(
        [
            "--input",
            str(source),
            "--scale",
            "2",
            "--preset",
            "photo",
            "--quality",
            "fast",
            "--format",
            "webp",
            "--json-progress",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    messages = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert messages[-1]["op"] == "done"
    assert messages[-1]["format"] == "webp"
    assert Path(str(messages[-1]["output"])).is_file()
