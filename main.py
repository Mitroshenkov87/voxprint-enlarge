"""Voxprint AI Enlarge.

    python main.py                         open the window
    python main.py --selftest              open the window, then quit (smoke test)
    python main.py --version
    python main.py --upscale FILE --out FILE [--scale 2|4] [--format avif|jxl|webp|jpeg|heic] [--quality N] [--lossless]
    python main.py --worker                internal: the inference process
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from enlarge.appinfo import APP_DISPLAY_NAME, version_line
from enlarge.batch import run_batch
from enlarge.paths import logs_dir
from enlarge.suite import gpu
from enlarge.ui import splash as splash_mod
from enlarge.ui.window import MainWindow
from voxprint_engine.formats import DEFAULT_FORMAT, DEFAULT_QUALITY
from voxprint_engine.hardware import assess_gpu
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


def _gui(argv: list[str]) -> int:
    running = QApplication.instance()
    app = running if isinstance(running, QApplication) else QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setOrganizationName("Voxprint")
    blocked = _gpu_block()
    if blocked:
        print(blocked, file=sys.stderr)
        QMessageBox.critical(None, APP_DISPLAY_NAME, blocked)
        return 1
    register_quietly()
    splash = splash_mod.show()
    splash.status("Loading", 0.35)
    window = MainWindow()
    splash.status("Opening", 0.85)
    window.show()
    splash.finish(window)
    if "--selftest" in argv:
        QTimer.singleShot(300, app.quit)
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _guard_stdio()
    if "--version" in args:
        print(version_line())
        return 0
    if "--worker" in args:
        return serve()
    _setup_logging()
    if "--upscale" in args:
        blocked = _gpu_block()
        if blocked:
            print(blocked, file=sys.stderr)
            return 1
        return _upscale(args)
    return _gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
