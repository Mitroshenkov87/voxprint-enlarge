"""Shared splash: version line, timing, progress, and the Qt import boundary."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication, QWidget

from voxprint_engine.ui.splash import (
    FADE_MS,
    MIN_VISIBLE_S,
    SplashScreen,
    choose_art,
    format_splash_version,
    resolve_build,
)

ROOT = Path(__file__).resolve().parent.parent


def _app() -> QApplication:
    running = QApplication.instance()
    if isinstance(running, QApplication):
        return running
    return QApplication([])


def _art(path: Path) -> None:
    Image.new("RGB", (32, 18), (12, 18, 28)).save(path, quality=85)


def test_version_line_reads_an_rc_build_and_a_final_build() -> None:
    rc = format_splash_version({"version": "0.2.0-rc.5", "codename": "Chayil"}, 12)
    assert rc == '0.2.0 RC 5 build 12 "Chayil"'
    final = format_splash_version({"version": "1.0.0", "codename": "Chayil"}, 12)
    assert final == '1.0.0 build 12 "Chayil"'
    bare = format_splash_version({"version": "1.0.0"}, 3)
    assert bare == "1.0.0 build 3"


def test_resolve_build_prefers_the_file_then_the_stamp_then_the_environment(tmp_path: Path) -> None:
    assert resolve_build({"version": "0.2.0-rc.5", "build": 12, "offset": 4}, tmp_path) == 12
    (tmp_path / "build_info.json").write_text(json.dumps({"build": 40}), encoding="utf-8")
    assert resolve_build({"version": "0.2.0-rc.5", "offset": 4}, tmp_path) == 40
    assert resolve_build({"offset": 4}, None, {"VOXPRINT_BUILD": "7"}) == 7
    assert resolve_build({"offset": 4}, None, {"GITHUB_RUN_NUMBER": "9"}) == 13
    assert resolve_build({"offset": 0}, None, {}) == 0


def test_choose_art_uses_the_2x_file_only_on_hidpi(tmp_path: Path) -> None:
    art = tmp_path / "splash.jpg"
    hidpi = tmp_path / "splash@2x.jpg"
    art.write_bytes(b"1x")
    assert choose_art(art, 2.0) == art
    hidpi.write_bytes(b"2x")
    assert choose_art(art, 2.0) == hidpi
    assert choose_art(art, 1.5) == art


def test_splash_text_comes_from_build_json(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    build = tmp_path / "BUILD.json"
    build.write_text(
        json.dumps({"version": "0.2.0-rc.5", "build": 12, "codename": "Chayil"}),
        encoding="utf-8",
    )
    splash = SplashScreen("Voxprint AI Enlarge", art, "#ff6b8a", build)
    assert splash.version_text == '0.2.0 RC 5 build 12 "Chayil"'
    splash.set_stage("Loading UI")
    assert splash._stage == "Loading UI"
    splash.close()


def test_finish_holds_the_minimum_then_fades(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    window = QWidget()
    splash = SplashScreen("Voxprint AI Enlarge", art, "#ff6b8a")
    started = time.monotonic()
    splash.show()
    splash.finish(window)
    elapsed = time.monotonic() - started
    assert elapsed >= MIN_VISIBLE_S + (FADE_MS / 1000) - 0.05
    assert elapsed < MIN_VISIBLE_S + (FADE_MS / 1000) + 0.8
    assert not splash.isVisible()
    window.close()


def test_finish_does_not_add_another_minimum_after_the_splash_has_waited(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    window = QWidget()
    splash = SplashScreen("Voxprint AI Enlarge", art, "#ff6b8a")
    splash.show()
    deadline = time.monotonic() + MIN_VISIBLE_S + 0.05
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    started = time.monotonic()
    splash.finish(window)
    elapsed = time.monotonic() - started
    assert elapsed < MIN_VISIBLE_S
    assert elapsed >= (FADE_MS / 1000) - 0.05
    window.close()


def test_set_progress_switches_from_indeterminate_to_determinate(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    splash = SplashScreen("Voxprint AI Enlarge", art, "#ff6b8a")
    splash.show()
    assert splash._fraction is None
    assert splash._timer.isActive()
    splash.set_progress(0.4)
    assert splash._fraction == 0.4
    assert not splash._timer.isActive()
    splash.set_progress(2)
    assert splash._fraction == 1.0
    splash.set_progress(None)
    assert splash._fraction is None
    assert splash._timer.isActive()
    splash.close()
    assert not splash._timer.isActive()


def test_context_manager_closes_when_finish_is_not_called(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    with SplashScreen("Voxprint AI Enlarge", art, "#ff6b8a") as splash:
        splash.set_stage("Loading registry")
        assert splash.isVisible()
    assert not splash.isVisible()
    assert not splash._timer.isActive()


def test_engine_and_headless_cli_do_not_import_pyside() -> None:
    script = """
import os, sys
sys.path.insert(0, os.environ["VOXPRINT_ROOT"])
import voxprint_engine
import voxprint_engine.api
import voxprint_engine.ui
from main import main
if main(["--version"]) != 0:
    raise SystemExit("version failed")
if "PySide6" in sys.modules:
    raise SystemExit("PySide6 was imported")
"""
    env = os.environ.copy()
    env["VOXPRINT_ROOT"] = str(ROOT)
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
