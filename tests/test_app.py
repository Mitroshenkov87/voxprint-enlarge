"""The window comes up offscreen, and a queued image survives a fake upscale."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PIL import Image
from PySide6.QtWidgets import QApplication, QLabel

from enlarge.ui.dialogs import CreditsDialog, SettingsDialog
from main import main
from voxprint_engine.hardware import NO_GPU_MESSAGE, OLD_DRIVER_MESSAGE, GpuInfo

from enlarge.appinfo import format_version, version_line
from enlarge.images import collect, plan_outputs
from enlarge.queue_state import dump_queue
from enlarge.platform_win import MIN_BUILD, START_MENU_FOLDER, check_os
from enlarge.ui.splash import BAR
from enlarge.ui.theme import CARD_GLASS, CONTRAST_PAIRS, MENU_BG, PRIMARY, ROOT_GLASS, TEXT, TEXT_FAINT, TEXT_MUTED, build_style
from enlarge.ui.window import STEPS, MainWindow


def _lum(hex_color: str) -> float:
    raw = hex_color.lstrip("#")
    channels = [int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted((_lum(foreground), _lum(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _app() -> QApplication:
    running = QApplication.instance()
    if isinstance(running, QApplication):
        return running
    return QApplication([])


@pytest.mark.parametrize("foreground,background", CONTRAST_PAIRS)
def test_theme_contrast(foreground: str, background: str) -> None:
    assert _ratio(foreground, background) >= 4.5


def _over_white(rgba: str) -> str:
    """Composite ``rgba(r,g,b,a)`` over white, the lightest desktop behind Acrylic."""
    red, green, blue, alpha = [int(part) for part in rgba[len("rgba("):-1].split(",")]
    channels = tuple(round(channel * alpha / 255 + 255 * (1 - alpha / 255)) for channel in (red, green, blue))
    return "#%02x%02x%02x" % channels


def test_theme_contrast_over_a_white_desktop() -> None:
    for glass in (ROOT_GLASS, CARD_GLASS):
        worst = _over_white(glass)
        for foreground in (TEXT, TEXT_MUTED, TEXT_FAINT):
            assert _ratio(foreground, worst) >= 4.5


def test_primary_actions_are_yellow_and_the_splash_stays_pink() -> None:
    css = build_style(False)
    assert PRIMARY == "#f5d76e"
    assert f"background: {PRIMARY}" in css
    assert MENU_BG in css
    assert BAR == "#ff6b8a"


def test_stylesheet_has_no_leftover_braces() -> None:
    css = build_style(True)
    assert "{{" not in css and "QPushButton" in css


def test_platform_names() -> None:
    assert START_MENU_FOLDER == "Voxprint"
    assert MIN_BUILD == 26100
    assert check_os(build=26100, is_windows=True).ok
    assert not check_os(build=22000, is_windows=True).ok
    assert "0.2.0-rc.1" in version_line()
    assert "pre-alpha" in format_version("0.1.0-pre")
    source = Path("enlarge/platform_win.py").read_text(encoding="utf-8")
    assert "22621" not in source
    assert "22H2" not in source


def test_collect_includes_subfolders(tmp_path: Path) -> None:
    root = tmp_path / "album"
    nested = root / "trip"
    nested.mkdir(parents=True)
    (root / "top.png").write_bytes(b"png")
    (nested / "deep.png").write_bytes(b"png")
    (nested / "clip.mp4").write_bytes(b"video")
    deep, warnings = collect([root], recursive=True)
    assert {path.name for path in deep} == {"top.png", "deep.png"}
    assert any("Clarity" in warning for warning in warnings)
    shallow, _quiet = collect([root], recursive=False)
    assert [path.name for path in shallow] == ["top.png"]


def test_collect_skips_video(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"not really")
    # collect checks the suffix and is_file, then plan/worker checks pixels.
    (tmp_path / "movie.mkv").write_bytes(b"video")
    found, warnings = collect([tmp_path], recursive=False)
    # The png is not a decodable image but the suffix is allowed at the picker.
    assert image in found
    assert any("Clarity" in warning for warning in warnings)
    planned = plan_outputs([image], None, 4, "avif", False)
    assert planned[0].name == "a_x4.avif"
    with pytest.raises(ValueError, match="PNG"):
        plan_outputs([image], None, 4, "png", False)


def test_window_starts_and_runs_a_sample(tmp_path: Path) -> None:
    _app()
    sample = tmp_path / "hill.png"
    Image.new("RGB", (16, 10), (180, 40, 40)).save(sample)
    window = MainWindow()
    assert [button.text() for button in window.step_buttons] == [f"{i}. {name}" for i, name in enumerate(STEPS, start=1)]
    assert window.stack.count() == 4
    window.add_paths([sample])
    assert len(window.sources) == 1
    window.output_page.set_scale(2)
    window.output_page.output_edit.setText(str(tmp_path / "out"))
    (tmp_path / "out").mkdir()
    window._on_output_changed()
    window.go(3)
    window.show()
    _app().processEvents()
    window.start_run()
    for _ in range(200):
        _app().processEvents()
        if window.job is None and window.rows and window.rows[0]["status"] == "Done":
            break
        time.sleep(0.05)
    else:
        status = window.rows[0]["status"] if window.rows else "no rows"
        message = window.rows[0]["message"] if window.rows else ""
        raise AssertionError(f"upscale did not finish: {status} {message} {window.run_page.status.text()}")
    output = Path(str(window.rows[0]["dst"]))
    assert output.is_file()
    assert output.suffix == ".avif"
    assert window.output_page.image_format() == "avif"
    labels = [window.output_page.format_box.itemText(i) for i in range(window.output_page.format_box.count())]
    assert labels[0] == "AVIF"
    assert "PNG" not in labels
    assert any(label == "JPEG (legacy)" for label in labels)
    assert Image.open(output).size == (32, 20)
    window.run_page.compare.set_pair(sample, output)
    assert not window.run_page.compare.before.isNull()
    assert not window.run_page.compare.after.isNull()
    window.run_page.compare.zoom_by(2)
    assert window.run_page.compare.zoom == 2.0
    window.run_page.compare.zoom_fit()
    assert window.run_page.compare.zoom == 1.0
    dialog = CreditsDialog(window, list(window.catalog.models), [])
    body = dialog.findChild(QLabel, "creditsBody")
    assert body is not None
    assert "Philip Hofmann" in body.text()
    assert "CC-BY-4.0" in body.text()
    assert "BSD-3-Clause" in body.text()
    assert "RealESRGAN_x4plus" in body.text()
    dialog.close()
    settings_dialog = SettingsDialog(window, window.settings)
    setting_labels = [settings_dialog.format_box.itemText(i) for i in range(settings_dialog.format_box.count())]
    assert setting_labels[0] == "AVIF"
    assert settings_dialog.format_box.currentData() == "avif"
    gpu_labels = [settings_dialog.gpu_box.itemText(i) for i in range(settings_dialog.gpu_box.count())]
    assert gpu_labels == ["Auto", "CUDA:0", "CUDA:1"]
    assert "PNG" not in setting_labels
    assert any(label == "JPEG (legacy)" for label in setting_labels)
    jxl = settings_dialog.format_box.findData("jxl")
    settings_dialog.format_box.setCurrentIndex(jxl)
    settings_dialog.lossless.setChecked(True)
    settings_dialog._save()
    assert settings_dialog.result_settings["format"] == "jxl"
    assert settings_dialog.result_settings["lossless"] is True
    settings_dialog.close()
    window.close()


def test_a_crashed_queue_is_restored(tmp_path: Path) -> None:
    _app()
    image = tmp_path / "hill.png"
    Image.new("RGB", (8, 6), (9, 9, 9)).save(image)
    done = image.with_name("hill_x4.avif")
    done.write_bytes(b"keep")
    pending = image.with_name("hill-b.png")
    Image.new("RGB", (8, 6), (3, 3, 3)).save(pending)
    dump_queue(
        [
            {"id": "done", "src": str(image), "dst": str(done), "status": "Done", "message": "ok"},
            {"id": "open", "src": str(pending), "dst": str(pending.with_name("hill-b_x4.avif")), "status": "Running", "message": ""},
        ]
    )
    window = MainWindow()
    statuses = {str(row["status"]) for row in window.rows}
    assert "Done" in statuses
    assert "Interrupted" in statuses
    window.close()


def test_main_selftest_and_version() -> None:
    assert main(["--version"]) == 0
    _app()
    assert main(["--selftest"]) == 0


def test_cli_upscale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "in.png"
    dest = tmp_path / "out.webp"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
    code = main(["--upscale", str(source), "--out", str(dest), "--scale", "4", "--format", "webp", "--quality", "80"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert dest.is_file()
    assert Image.open(dest).size == (32, 32)


def test_cli_refuses_without_a_cuda_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.setattr("voxprint_engine.hardware.query_gpus", lambda runner=None: [])
    source = tmp_path / "in.png"
    dest = tmp_path / "out.avif"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
    code = main(["--upscale", str(source), "--out", str(dest), "--scale", "2"])
    captured = capsys.readouterr()
    assert code == 1
    assert NO_GPU_MESSAGE in captured.err
    assert not dest.exists()


def test_cli_refuses_a_driver_below_600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.setattr(
        "voxprint_engine.hardware.query_gpus",
        lambda runner=None: [GpuInfo("NVIDIA GeForce RTX 4090", "560.35.03", (8, 9))],
    )
    source = tmp_path / "in.png"
    dest = tmp_path / "out.avif"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(source)
    code = main(["--upscale", str(source), "--out", str(dest), "--scale", "2"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == OLD_DRIVER_MESSAGE.format(driver="560.35.03")
    assert not dest.exists()


def test_gui_refuses_without_a_cuda_gpu(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.setattr("voxprint_engine.hardware.query_gpus", lambda runner=None: [])
    shown: list[str] = []

    def critical(_parent: object, title: str, text: str) -> None:
        shown.append(f"{title}: {text}")

    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", critical)
    _app()
    code = main([])
    captured = capsys.readouterr()
    assert code == 1
    assert NO_GPU_MESSAGE in captured.err
    assert shown == [f"Voxprint AI Enlarge: {NO_GPU_MESSAGE}"]
