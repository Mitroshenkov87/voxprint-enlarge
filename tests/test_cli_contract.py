"""Headless CLI contract: exit codes, dry-run, warm server, parent lock, long side, no window."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PIL import Image
from PySide6.QtWidgets import QApplication

from enlarge.ipc.discovery import known_install_exe
from enlarge.ipc.jobs import resolve_model_id
from enlarge.ipc.protocol import ERROR_CODES, exit_code
from enlarge.ipc.scale import choose_scale, fit_long_side, planned_size
from enlarge.ipc.server import IpcServer
from enlarge.ui.icons import suite_icon
from main import main
from voxprint_engine.gpu_lock import lock_path, read_lock
from voxprint_engine.worker import WorkerError

ROOT = Path(__file__).resolve().parent.parent


def _app() -> QApplication:
    running = QApplication.instance()
    if isinstance(running, QApplication):
        return running
    return QApplication([])


def _picture(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    Image.new("RGB", size, (12, 40, 90)).save(path)


def _lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _job(source: Path, *extra: str) -> list[str]:
    return [
        "--input",
        str(source),
        "--preset",
        "photo",
        "--json-progress",
        *extra,
    ]


def test_exit_code_map() -> None:
    assert exit_code("internal") == 1
    assert exit_code("bad_input") == 2
    assert exit_code("unsupported_format") == 3
    assert exit_code("gpu_unsupported") == 4
    assert exit_code("gpu_busy") == 5
    assert exit_code("oom") == 6
    assert exit_code("model_missing") == 7
    assert exit_code("io_error") == 8
    assert exit_code("cancelled") == 130
    assert set(ERROR_CODES) == {
        "internal",
        "bad_input",
        "unsupported_format",
        "gpu_unsupported",
        "gpu_busy",
        "oom",
        "model_missing",
        "io_error",
        "cancelled",
    }


def test_long_side_maths() -> None:
    assert choose_scale(8, 6, 16) == 2
    assert choose_scale(8, 6, 17) == 4
    assert choose_scale(8, 6, 32) == 4
    with pytest.raises(ValueError):
        choose_scale(8, 6, 33)
    assert planned_size(8, 6, 4, 20) == (20, 15)
    assert fit_long_side(16, 12, 10) == (10, 8)
    assert planned_size(8, 6, 2, 0) == (16, 12)


def test_dry_run_prints_the_plan_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    called = {"worker": False}

    def _worker() -> None:
        called["worker"] = True
        raise AssertionError("dry-run loaded a worker")

    monkeypatch.setattr("enlarge.ipc.jobs.WorkerClient", _worker)
    code = main(_job(source, "--scale", "2", "--format", "webp", "--dry-run"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert called["worker"] is False
    messages = _lines(captured.out)
    assert [message["type"] for message in messages] == ["accepted", "done"]
    assert messages[-1]["width"] == 16
    assert messages[-1]["height"] == 12
    assert messages[-1]["format"] == "webp"
    planned = Path(str(messages[-1]["output"]))
    assert planned.name == "hill.enlarged-2x.webp"
    assert not planned.exists()
    assert all(message["protocol"] == 1 for message in messages)


def test_long_side_dry_run_and_real_resize(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    code = main(_job(source, "--long-side", "20", "--format", "webp", "--dry-run"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    plan = _lines(captured.out)[-1]
    assert plan["width"] == 20 and plan["height"] == 15
    assert Path(str(plan["output"])).name == "hill.enlarged-4x.webp"
    assert not Path(str(plan["output"])).exists()
    code = main(_job(source, "--long-side", "20", "--format", "webp"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    done = _lines(captured.out)[-1]
    output = Path(str(done["output"]))
    assert output.is_file()
    assert Image.open(output).size == (20, 15)
    too_far = main(_job(source, "--long-side", "100", "--dry-run"))
    captured = capsys.readouterr()
    assert too_far == 2
    assert _lines(captured.out)[-1]["code"] == "bad_input"


def test_argument_format_model_gpu_and_io_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    assert main(["--input", str(source), "--preset", "photo"]) == 2
    assert main(_job(source, "--scale", "3")) == 2
    both = main(_job(source, "--scale", "2", "--long-side", "20"))
    assert both == 2
    capsys.readouterr()
    png = main(_job(source, "--scale", "2", "--format", "png", "--dry-run"))
    assert png == 3
    assert _lines(capsys.readouterr().out)[-1]["code"] == "unsupported_format"
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS")
    missing = main(_job(source, "--scale", "2", "--dry-run"))
    assert missing == 7
    assert _lines(capsys.readouterr().out)[-1]["code"] == "model_missing"
    monkeypatch.setattr("enlarge.ipc.jobs.weight_report", lambda *_args, **_kwargs: ("ready", "sha256 matches"))
    monkeypatch.setattr("enlarge.ipc.capabilities.query_gpus", lambda runner=None: [])
    gpu = main(_job(source, "--scale", "2", "--dry-run"))
    assert gpu == 4
    assert _lines(capsys.readouterr().out)[-1]["code"] == "gpu_unsupported"
    monkeypatch.setenv("VOXPRINT_CI_LANCZOS", "1")
    blocker = tmp_path / "not-dir"
    blocker.write_text("x", encoding="utf-8")
    io = main(_job(source, "--scale", "2", "--output", str(blocker / "out.webp"), "--dry-run"))
    assert io == 8
    assert _lines(capsys.readouterr().out)[-1]["code"] == "io_error"


def test_oom_and_internal_from_the_worker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hill.png"
    _picture(source)

    class _Boom:
        def __init__(self, text: str) -> None:
            self.text = text

        def __enter__(self) -> _Boom:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def upscale(self, *_args: object, **_kwargs: object) -> None:
            raise WorkerError(self.text)

        def interrupt(self) -> None:
            return None

    monkeypatch.setattr("enlarge.ipc.jobs.WorkerClient", lambda: _Boom("out of memory"))
    assert main(_job(source, "--scale", "2")) == 6
    assert _lines(capsys.readouterr().out)[-1]["code"] == "oom"
    monkeypatch.setattr("enlarge.ipc.jobs.WorkerClient", lambda: _Boom("something else"))
    assert main(_job(source, "--scale", "2")) == 1
    assert _lines(capsys.readouterr().out)[-1]["code"] == "internal"


def test_gpu_busy_reports_the_lock_holder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock_path().write_text(
        json.dumps(
            {"owner": "canvas", "pid": os.getpid(), "job": "canvas:export", "started": future, "eta": future}
        ),
        encoding="utf-8",
    )
    code = main(_job(source, "--scale", "2", "--lock-timeout", "0.2"))
    captured = capsys.readouterr()
    assert code == 5, captured.err
    messages = _lines(captured.out)
    waiting = next(message for message in messages if message.get("stage") == "waiting_gpu")
    assert waiting["lock_holder"] == {"owner": "canvas", "pid": os.getpid(), "job": "canvas:export"}
    assert messages[-1]["code"] == "gpu_busy"
    assert messages[-1]["retryable"] is True
    assert not (tmp_path / "hill.enlarged-2x.avif").exists()


def test_parent_lock_is_shared_and_not_released(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock_path().write_text(
        json.dumps({"owner": "canvas", "pid": os.getpid(), "job": "canvas:export", "started": future, "eta": future}),
        encoding="utf-8",
    )
    dest = tmp_path / "out.webp"
    code = main(
        _job(source, "--scale", "2", "--format", "webp", "--output", str(dest), "--parent-lock-pid", str(os.getpid()))
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert dest.is_file()
    held = read_lock()
    assert held is not None
    assert held["owner"] == "canvas"
    assert held["pid"] == os.getpid()
    mismatch = main(_job(source, "--scale", "2", "--dry-run", "--parent-lock-pid", "424242"))
    assert mismatch == 2
    assert _lines(capsys.readouterr().out)[-1]["code"] == "bad_input"
    assert read_lock() is not None and read_lock()["owner"] == "canvas"


def test_upscale_subcommand_and_legacy_quality(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    code = main(["upscale", *_job(source, "--scale", "2", "--format", "webp", "--quality-tier", "max", "--quality", "80")])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    done = _lines(captured.out)[-1]
    assert done["type"] == "done"
    assert done["format"] == "webp"
    assert Path(str(done["output"])).is_file()


def test_contract_does_not_open_a_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hill.png"
    _picture(source)

    class _Window:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("the CLI opened a window")

    monkeypatch.setattr("enlarge.ui.window.MainWindow", _Window)
    monkeypatch.setattr("enlarge.ipc.server.MainWindow", _Window)
    code = main(_job(source, "--scale", "2", "--format", "webp"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _lines(captured.out)[-1]["type"] == "done"


def _call_while_serving(argv: list[str]) -> int:
    """Run ``main`` off the Qt thread so a blocking socket can be served by ``processEvents``."""
    box: dict[str, int] = {}

    def _target() -> None:
        box["code"] = main(argv)

    thread = threading.Thread(target=_target)
    thread.start()
    app = _app()
    deadline = time.monotonic() + 30
    while thread.is_alive() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1)
    assert not thread.is_alive()
    return int(box["code"])


def test_warm_server_forwards_and_matches_one_shot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    _app()
    server = IpcServer()
    assert server.listen()
    warm_out = tmp_path / "warm.webp"
    try:
        code = _call_while_serving(_job(source, "--scale", "2", "--format", "webp", "--output", str(warm_out)))
    finally:
        server.close()
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert server.window is None
    warm = _lines(captured.out)
    assert warm[0]["type"] == "accepted"
    assert warm[-1]["type"] == "done"
    assert warm[-1]["width"] == 16 and warm[-1]["height"] == 12
    assert all(message["protocol"] == 1 and message["type"] for message in warm)
    cold_out = tmp_path / "cold.webp"
    code = main(_job(source, "--scale", "2", "--format", "webp", "--output", str(cold_out)))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    cold = _lines(captured.out)
    assert cold[-1]["width"] == warm[-1]["width"]
    assert cold[-1]["height"] == warm[-1]["height"]
    assert cold[-1]["format"] == warm[-1]["format"]
    assert warm_out.is_file() and cold_out.is_file()


def test_capabilities_go_through_the_warm_server(capsys: pytest.CaptureFixture[str]) -> None:
    _app()
    server = IpcServer()
    assert server.listen()
    try:
        code = _call_while_serving(["--capabilities", "--json"])
    finally:
        server.close()
    captured = capsys.readouterr()
    assert code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["app"] == "enlarge"
    assert payload["protocol"] == 1
    assert "op" not in payload
    assert "type" not in payload
    local = main(["--capabilities", "--json"])
    assert local == 0
    again = json.loads(capsys.readouterr().out)
    assert again["app"] == payload["app"]
    assert again["gpu"] == payload["gpu"]
    assert again["models"] == payload["models"]


@pytest.mark.skipif(sys.platform == "win32", reason="TerminateProcess does not run the Python SIGTERM handler")
def test_sigterm_removes_the_partial_output(tmp_path: Path) -> None:
    source = tmp_path / "in.png"
    _picture(source)
    env = os.environ.copy()
    env["VOXPRINT_FAKE_DELAY"] = "3"
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py"), *_job(source, "--scale", "2", "--format", "webp")],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    target = tmp_path / "in.enlarged-2x.webp"
    deadline = time.monotonic() + 15
    while proc.poll() is None and not target.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert target.exists()
    proc.send_signal(signal.SIGTERM)
    code = proc.wait(timeout=20)
    assert code == 130
    assert not target.exists()


def test_known_install_exe_and_icons() -> None:
    path = known_install_exe()
    if sys.platform == "win32":
        assert path.name == "VoxprintEnlarge.exe"
        assert "Voxprint" in path.parts and "Enlarge" in path.parts
    else:
        assert path.name == "voxprint-enlarge"
        assert path.parts[-3:] == ("voxprint", "enlarge", "voxprint-enlarge")
    _app()
    icon = suite_icon("save")
    assert not icon.isNull()
    assert suite_icon("export").isNull() is False
    assert suite_icon("download").isNull() is False
    assert suite_icon("save-as").isNull() is False
    assert suite_icon("save-copy").isNull() is False


def test_omitted_preset_defaults_to_photo_fast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    seen: dict[str, tuple[str, str, str]] = {}
    real = resolve_model_id

    def _spy(preset: str, quality: str, model: str) -> str:
        seen["args"] = (preset, quality, model)
        return real(preset, quality, model)

    monkeypatch.setattr("enlarge.ipc.jobs.resolve_model_id", _spy)
    code = main(
        [
            "--input",
            str(source),
            "--scale",
            "2",
            "--format",
            "webp",
            "--dry-run",
            "--json-progress",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _lines(captured.out)[-1]["type"] == "done"
    assert seen["args"] == ("photo", "fast", "")


def test_headless_commands_do_not_import_pyside(tmp_path: Path) -> None:
    source = tmp_path / "hill.png"
    _picture(source)
    script = """
import json, os, sys
sys.path.insert(0, os.environ["VOXPRINT_ROOT"])
from main import main
source = os.environ["VOXPRINT_SOURCE"]
checks = [
    ["--version"],
    ["--capabilities", "--json"],
    ["--selftest"],
    ["--selftest", "--json"],
    ["--input", source, "--dry-run", "--json-progress"],
    ["upscale", "--input", source, "--scale", "2", "--dry-run", "--json-progress"],
]
for argv in checks:
    rc = main(argv)
    if rc != 0:
        raise SystemExit(f"{argv} exited {rc}")
if "PySide6" in sys.modules:
    raise SystemExit("PySide6 was imported: " + ", ".join(name for name in sys.modules if name.startswith("PySide6")))
"""
    env = os.environ.copy()
    env["VOXPRINT_ROOT"] = str(ROOT)
    env["VOXPRINT_SOURCE"] = str(source)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
