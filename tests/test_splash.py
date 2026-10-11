"""Splash timing and the Qt import boundary. The window comes from voxprint_common."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

from voxprint_common.buildinfo import label, parse
from voxprint_common.ui.splash import FADE_MS, HEIGHT, MIN_VISIBLE_S, WIDTH, Splash, SplashEvent

ROOT = Path(__file__).resolve().parent.parent


def _app() -> QApplication:
    running = QApplication.instance()
    if isinstance(running, QApplication):
        return running
    return QApplication([])


def _art(path: Path) -> None:
    Image.new("RGB", (32, 18), (12, 18, 28)).save(path, quality=85)


def _info(**extra: object):
    payload: dict[str, object] = {"version": "0.2.0-rc.6", "codename": "Emunah", "build": 12}
    payload.update(extra)
    return parse(payload, app="enlarge", env={})


def test_label_reads_an_rc_build_and_a_final_build() -> None:
    rc = label(parse({"version": "0.2.0-rc.6", "codename": "Emunah", "build": 12}, app="enlarge", env={}))
    assert rc == '0.2.0 RC 6 build 12 "Emunah"'
    final = label(parse({"version": "1.0.0", "codename": "Emunah", "build": 12}, app="enlarge", env={}))
    assert final == '1.0.0 build 12 "Emunah"'


def test_a_missing_build_is_dev_not_zero() -> None:
    text = label(parse({"version": "0.2.0-rc.6", "codename": "Emunah"}, app="enlarge", env={}))
    assert text == '0.2.0 RC 6 dev "Emunah"'
    assert "build 0" not in text


def test_click_uses_the_public_hook(tmp_path: Path) -> None:
    _app()
    seen: list[SplashEvent] = []
    art = tmp_path / "splash.jpg"
    _art(art)
    splash = Splash(art, "#ff6b8a", _info(), on_click=seen.append)
    splash.mouse_press(SplashEvent(3, 4, "left"))
    assert seen == [SplashEvent(3, 4, "left")]
    splash.finish()


def test_layout_is_the_suite_splash() -> None:
    assert WIDTH == 960
    assert HEIGHT == 540


def test_splash_text_comes_from_build_json(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    splash = Splash(art, "#ff6b8a", _info())
    assert label(splash.build_info) == '0.2.0 RC 6 build 12 "Emunah"'
    splash.set_stage("Loading UI")
    assert splash._stage == "Loading UI"
    splash.finish()


def test_finish_holds_the_minimum_then_fades(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    splash = Splash(art, "#ff6b8a", _info())
    started = time.monotonic()
    splash.finish()
    elapsed = time.monotonic() - started
    assert elapsed >= MIN_VISIBLE_S + (FADE_MS / 1000) - 0.05
    assert elapsed < MIN_VISIBLE_S + (FADE_MS / 1000) + 1.2


def test_finish_does_not_add_another_minimum_after_the_splash_has_waited(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    splash = Splash(art, "#ff6b8a", _info())
    deadline = time.monotonic() + MIN_VISIBLE_S + 0.05
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    started = time.monotonic()
    splash.finish()
    elapsed = time.monotonic() - started
    assert elapsed < MIN_VISIBLE_S
    assert elapsed >= (FADE_MS / 1000) - 0.05


def test_set_progress_switches_from_indeterminate_to_determinate(tmp_path: Path) -> None:
    _app()
    art = tmp_path / "splash.jpg"
    _art(art)
    splash = Splash(art, "#ff6b8a", _info())
    assert splash._fraction is None
    splash.set_progress(0.4)
    assert splash._fraction == 0.4
    try:
        splash.set_progress(2)
    except ValueError:
        pass
    else:
        raise AssertionError("a fraction above 1 must be refused")
    splash.finish()


def test_engine_and_headless_cli_do_not_import_pyside() -> None:
    script = """
import os, sys
sys.path.insert(0, os.environ["VOXPRINT_ROOT"])
import voxprint_engine
import voxprint_engine.api
from main import main
if main(["--version"]) != 0:
    raise SystemExit("version failed")
loaded = [name for name in sys.modules if name.startswith("PySide6") or name.startswith("voxprint_engine.ui") or name == "voxprint_engine.shared_manifest"]
if loaded:
    raise SystemExit("suite modules were imported: " + ", ".join(loaded))
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
