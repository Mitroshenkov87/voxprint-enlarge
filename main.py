"""Voxprint AI Enlarge.

    python main.py                         open the window
    python main.py --selftest [--json]     re-check the shared install and repair it
    python main.py --version
    python main.py --capabilities --json   one JSON object, exit 0 with or without a GPU
    python main.py --serve --hidden        protocol 1 listener, no window until open_ui
    python main.py --register [--exe PATH] discovery records for Lucent
    python main.py --unregister            remove those discovery records
    python main.py --install-suite         shared Python, CUDA wheels, and the default model
    python main.py --uninstall-suite       drop this app's hold on those shared files
    python main.py --install-gpu           torch, CUDA 13 ONNX Runtime, and TensorRT wheels
    python main.py --input FILE --scale 2|4 --preset photo|art [--model ID] [--quality fast|max] [--format FMT] [--output PATH] --json-progress
    python main.py --upscale FILE --out FILE [--scale 2|4] [--format avif|jxl|webp|jpeg|heic] [--quality N] [--lossless]
    python main.py --worker                internal: the inference process
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from enlarge.appinfo import APP_DISPLAY_NAME, version_line
from enlarge.batch import run_batch
from enlarge.cli import run_capabilities, run_contract
from enlarge.cli_parser import parse_cli
from enlarge.ipc.register import register_install, unregister_install
from enlarge.paths import logs_dir
from enlarge.suite import gpu
from voxprint_engine.formats import DEFAULT_FORMAT, DEFAULT_QUALITY
from voxprint_engine.hardware import assess_gpu, ci_lanczos
from voxprint_engine.suite_runtime import (
    RuntimeReport,
    SuiteInstallError,
    ensure_runtime,
    install_gpu_wheels,
    release_runtime,
)
from voxprint_engine.registry import DEFAULT_MODEL_ID, load_builtin
from voxprint_engine.store import register_quietly
from voxprint_engine.worker import UpscaleRequest
from voxprint_engine.worker.server import serve

_USAGE = (
    "usage: main.py --upscale FILE --out FILE [--scale 2|4] "
    "[--format avif|jxl|webp|jpeg|heic] [--quality N] [--lossless] [--model ID] [--double]"
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arg(argv: list[str], name: str, default: str | None = None) -> str | None:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
        return argv[index + 1]
    return default


def _guard_stdio() -> None:
    """pythonw.exe on Windows can start with stdout and stderr set to None."""
    if sys.platform != "win32":
        return
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _setup_logging() -> None:
    try:
        logging.basicConfig(
            filename=str(logs_dir() / "enlarge.log"),
            level=logging.INFO,
            encoding="utf-8",
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    except OSError:
        return


def _gpu_block() -> str:
    """Empty when the machine may run. Otherwise the sentence to show and exit on."""
    return assess_gpu().message


def _upscale(argv: list[str]) -> int:
    source = _arg(argv, "--upscale")
    dest = _arg(argv, "--out")
    if not source or not dest:
        print(_USAGE, file=sys.stderr)
        return 2
    try:
        scale = int(_arg(argv, "--scale", "4") or "4")
        quality = int(_arg(argv, "--quality", str(DEFAULT_QUALITY)) or str(DEFAULT_QUALITY))
    except ValueError:
        print("scale and quality must be integers", file=sys.stderr)
        return 2
    image_format = _arg(argv, "--format", DEFAULT_FORMAT) or DEFAULT_FORMAT
    model_id = _arg(argv, "--model", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID
    catalog = load_builtin(release=True)
    model = catalog.get(model_id)
    if model is None or not model.selectable:
        print(f"unknown or unavailable model {model_id}", file=sys.stderr)
        return 2
    register_quietly()
    request = UpscaleRequest(
        id="cli",
        src=str(Path(source)),
        dst=str(Path(dest)),
        scale=scale,
        format=image_format,
        quality=quality,
        model_id=model.id,
        tile=0,
        model_tile=model.tile,
        double="--double" in argv,
        device=gpu(),
        lossless="--lossless" in argv,
    )
    failed: list[str] = []

    def on_item(_job_id: str, ok: bool, message: str, path: str) -> None:
        if ok:
            print(path or message)
        else:
            failed.append(message)
            print(message, file=sys.stderr)

    run_batch([request], on_progress=lambda *_parts: None, on_item=on_item, cancel=lambda: False)
    return 1 if failed else 0


def _failure_text(report: RuntimeReport) -> str:
    """One line naming each component that is still missing or failed."""
    parts = [f"{item.id}: {item.detail}" for item in report.components if item.state in {"failed", "missing"}]
    return "; ".join(parts) or "shared suite install is incomplete"


def _ensure_shared(progress: Callable[[str], None] | None = None) -> str:
    """Empty when the shared install is ready. CI skips the check."""
    if ci_lanczos():
        return ""
    sink = progress or (lambda message: print(message, file=sys.stderr, flush=True))
    try:
        report = ensure_runtime("enlarge", progress=sink)
    except SuiteInstallError as exc:
        return str(exc)
    if report.ok:
        return ""
    return _failure_text(report)


def _refuse_if_broken() -> int:
    """Repair the shared install. Return 1 when it is still incomplete."""
    message = _ensure_shared()
    if not message:
        return 0
    print(message, file=sys.stderr)
    return 1


def _print_suite(report: RuntimeReport, lines: list[str], as_json: bool, *, done: str) -> None:
    """Human lines on stdout, or one JSON object. Progress already collected stays on stderr for JSON."""
    if as_json:
        if lines:
            print("\n".join(lines), file=sys.stderr)
        print(json.dumps(report.as_dict()))
        return
    for line in lines:
        print(line, flush=True)
    if report.skipped:
        print("shared suite: skipped")
        return
    if report.ok:
        print(done)
        return
    print("shared suite: incomplete", file=sys.stderr)
    for item in report.components:
        if item.state in {"failed", "missing"}:
            print(f"{item.id}: {item.detail}", file=sys.stderr)


def _selftest(as_json: bool) -> int:
    """Re-check the shared install without opening a window."""
    lines: list[str] = []
    try:
        report = ensure_runtime("enlarge", progress=lines.append)
    except SuiteInstallError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_suite(report, lines, as_json, done="shared suite: ready")
    return 0 if report.ok else 1


def _install_suite() -> int:
    """Install shared Python, the CUDA wheels, and the default model."""
    lines: list[str] = []
    try:
        report = ensure_runtime("enlarge", progress=lines.append)
    except SuiteInstallError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_suite(report, lines, as_json=False, done="shared suite: ready")
    return 0 if report.ok else 1


def _uninstall_suite() -> int:
    """Drop Enlarge's hold on the shared runtime. Delete a file only at zero holders."""
    lines: list[str] = []
    try:
        report = release_runtime("enlarge", progress=lines.append)
    except SuiteInstallError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_suite(report, lines, as_json=False, done="shared suite: released")
    return 0 if report.ok else 1


def _gui(_argv: list[str]) -> int:
    # Qt is imported on the window path only. Headless commands must not load libEGL.
    from PySide6.QtWidgets import QApplication, QMessageBox

    from enlarge.ui import splash as splash_mod
    from enlarge.ui.window import MainWindow

    running = QApplication.instance()
    app = running if isinstance(running, QApplication) else QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setOrganizationName("Voxprint")
    splash = splash_mod.show()
    if not ci_lanczos():
        seen = {"n": 0}

        def on_progress(message: str) -> None:
            seen["n"] += 1
            splash.status(message, min(0.35, 0.05 * seen["n"]))
            print(message, file=sys.stderr, flush=True)

        splash.status("Checking the shared install", 0.08)
        message = _ensure_shared(on_progress)
        if message:
            splash.close()
            print(message, file=sys.stderr)
            QMessageBox.critical(None, APP_DISPLAY_NAME, message)
            return 1
    blocked = _gpu_block()
    if blocked:
        splash.close()
        print(blocked, file=sys.stderr)
        QMessageBox.critical(None, APP_DISPLAY_NAME, blocked)
        return 1
    register_quietly()
    splash.status("Loading", 0.45)
    window = MainWindow()
    splash.status("Opening", 0.9)
    window.show()
    splash.finish(window)
    return int(app.exec())


def _register(exe: str | None) -> int:
    register_install(Path(exe or str(ROOT / "main.py")))
    return 0


def _unregister() -> int:
    unregister_install()
    return 0


def _install_gpu() -> int:
    """Install torch and the CUDA 13 wheels into this interpreter."""
    try:
        install_gpu_wheels(sys.executable, lambda message: print(message, flush=True))
    except SuiteInstallError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _serve() -> int:
    # Qt is imported here so --capabilities and --dry-run never load libEGL.
    from PySide6.QtWidgets import QApplication

    from enlarge.ipc.server import IpcServer

    running = QApplication.instance()
    app = running if isinstance(running, QApplication) else QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setOrganizationName("Voxprint")
    server = IpcServer()
    if not server.listen():
        print(server.error or "could not listen for Lucent", file=sys.stderr)
        return 1
    return int(app.exec())


def main(argv: list[str] | None = None) -> int:
    """Dispatch one invocation. The parser in ``enlarge.cli_parser`` is the flag list."""
    args = list(sys.argv[1:] if argv is None else argv)
    _guard_stdio()
    parsed = parse_cli(args)
    if isinstance(parsed, int):
        return parsed
    if parsed.command == "upscale":
        _setup_logging()
        if not parsed.dry_run:
            broken = _refuse_if_broken()
            if broken:
                return broken
        return run_contract(args)
    if parsed.version:
        print(version_line())
        return 0
    if parsed.worker:
        return serve()
    if parsed.capabilities:
        if not parsed.json:
            print("usage: voxprint-enlarge --capabilities --json", file=sys.stderr)
            return 2
        return run_capabilities()
    if parsed.register:
        return _register(parsed.exe)
    if parsed.unregister:
        return _unregister()
    if parsed.install_gpu:
        return _install_gpu()
    if parsed.install_suite:
        return _install_suite()
    if parsed.uninstall_suite:
        return _uninstall_suite()
    if parsed.selftest:
        return _selftest(bool(parsed.json))
    _setup_logging()
    if parsed.serve:
        broken = _refuse_if_broken()
        if broken:
            return broken
        return _serve()
    if parsed.input or parsed.dry_run:
        if parsed.input and not parsed.dry_run:
            broken = _refuse_if_broken()
            if broken:
                return broken
        return run_contract(args)
    if parsed.upscale:
        broken = _refuse_if_broken()
        if broken:
            return broken
        blocked = _gpu_block()
        if blocked:
            print(blocked, file=sys.stderr)
            return 1
        return _upscale(args)
    return _gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
