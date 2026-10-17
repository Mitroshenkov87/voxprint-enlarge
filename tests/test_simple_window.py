"""The shared simple screen: one settings state, registry ids, and the wipe."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QLabel, QPushButton, QSlider, QSpinBox

from enlarge.cli_parser import build_parser
from enlarge.ipc.capabilities import ART, PHOTO_FAST, PHOTO_MAX
from enlarge.ipc.server import MainWindow
from enlarge.ui.jobplan import JobSettings, command_line, model_scale
from enlarge.ui.preset_map import map_path, picture_map
from enlarge.ui.simple_window import (
    SimpleWindow,
    known_model_ids,
    preset_model_ids,
    registry_models,
)
from voxprint_common.i18n import set_language
from voxprint_common.ui.presets import PresetMatch
from voxprint_common.ui.simple_image import Phase
from voxprint_common.ui.variants import InputKind, ResultKind, VariantId, surface_for

_ROOT = Path(__file__).resolve().parents[1]


def _app() -> QApplication:
    running = QApplication.instance()
    if isinstance(running, QApplication):
        return running
    return QApplication([])


def test_presets_name_registry_ids_from_one_map() -> None:
    ids = {model.id for model in registry_models()}
    written = preset_model_ids()
    assert set(written) == {"photographs", "detail", "drawings"}
    assert set(written.values()) == known_model_ids()
    assert written["photographs"] == PHOTO_FAST
    assert written["detail"] == PHOTO_MAX
    assert written["drawings"] == ART
    assert set(written.values()) <= ids
    for model_id in written.values():
        assert not model_id.endswith((".pth", ".onnx", ".pt"))
        assert "/" not in model_id
    window_text = (_ROOT / "enlarge" / "ui" / "simple_window.py").read_text(encoding="utf-8")
    preset_text = (_ROOT / "enlarge" / "ui" / "presets" / "enlarge.json").read_text(encoding="utf-8")
    for model_id in written.values():
        assert model_id not in window_text
        assert model_id not in preset_text
        assert model_id in map_path().read_text(encoding="utf-8")
    assert {row.ipc_preset for row in picture_map()} == {"photo", "art"}


def test_catalogs_share_keys() -> None:
    folder = _ROOT / "enlarge" / "ui" / "locales"
    catalogs = [json.loads((folder / name).read_text(encoding="utf-8")) for name in ("en.json", "ru.json", "de.json")]
    assert set(catalogs[0]) == set(catalogs[1]) == set(catalogs[2])


def test_preset_edit_reset_and_mode_keep_one_state() -> None:
    _app()
    set_language("en")
    window = SimpleWindow()
    assert window.simple.get("enlarge.model") == preset_model_ids()["photographs"]
    assert window.simple.get("enlarge.scale") == "4"
    assert window.simple.get("enlarge.format") == "avif"
    assert window.simple.get("enlarge.quality") == 60
    assert window.simple.get("enlarge.lossless") is False
    window.simple.choose("filetype", "jxl-lossless")
    assert window.simple.get("enlarge.format") == "jxl"
    assert window.simple.get("enlarge.lossless") is True
    window.simple.choose("filetype", "avif")
    assert window.simple.get("enlarge.lossless") is False
    before = window.simple.snapshot()
    window.view.set_expert(True)
    assert window.simple.snapshot() == before
    window.view.set_expert(False)
    assert window.simple.snapshot() == before
    window.view.set_expert(True)
    window.simple.set_value("enlarge.quality", 80)
    status = window.simple.preset_status("pictures")
    assert status is not None
    assert status.kind is PresetMatch.MODIFIED
    assert status.option_id == "photographs"
    window.view.refresh()
    chip = window.view.root.findChild(QPushButton, "chip-pictures-photographs")
    assert chip is not None
    assert chip.text() == "Photographs + changes"
    reset = window.view.root.findChild(QPushButton, "expertReset")
    assert reset is not None and reset.text() == "Reset to Photographs"
    window.simple.reset_group("pictures")
    assert window.simple.get("enlarge.model") == preset_model_ids()["photographs"]
    assert window.simple.get("enlarge.quality") == 60
    restored = window.simple.preset_status("pictures")
    assert restored is not None and restored.kind is PresetMatch.EXACT
    window.close()


def test_builtin_enlarge_surface_shows_a_picture_and_the_old_window_stays_unlaunched() -> None:
    _app()
    assert MainWindow is SimpleWindow
    window = SimpleWindow()
    assert window.surface is surface_for("enlarge")
    assert window.surface.variant is VariantId.IMAGE
    assert window.surface.input_kind is InputKind.FILE
    assert window.surface.result_kind is ResultKind.PICTURE
    window.view.set_prompt_text("picture.png")
    assert window.simple.generate()
    assert window.simple.phase is Phase.BUSY
    assert window.simple.finish("result-1") is not None
    window.view.refresh()
    picture = window.view.root.findChild(QLabel, "result")
    assert picture is not None
    pixmap = picture.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    window.close()


def grab_window(window: SimpleWindow, path: Path) -> None:
    """Save a 1200 by 900 grab of ``window``."""
    window.resize(1200, 900)
    window.show()
    app = _app()
    for _ in range(4):
        app.processEvents()
    window.resize(1200, 900)
    app.processEvents()
    image = window.grab()
    if image.size() != QSize(1200, 900):
        raise RuntimeError(f"{path.name} is {image.width()} by {image.height()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(path)):
        raise RuntimeError(f"could not write {path}")


def test_grabs(tmp_path: Path) -> None:
    _app()
    shots = {
        "en-simple.png": ("en", False, False),
        "en-expert.png": ("en", True, False),
        "en-result.png": ("en", False, True),
        "ru-simple.png": ("ru", False, False),
        "ru-expert.png": ("ru", True, False),
        "ru-result.png": ("ru", False, True),
    }
    for name, (lang, expert, wipe) in shots.items():
        set_language(lang)
        window = SimpleWindow()
        if expert:
            window.view.set_expert(True)
        if wipe:
            window.view.set_prompt_text("picture.png")
            assert window.simple.generate()
            assert window.simple.finish(f"shot-{lang}") is not None
            window.view.refresh()
        grab_window(window, tmp_path / name)
        window.close()
    set_language("en")


def test_every_cli_flag_is_in_the_feature_map() -> None:
    text = (_ROOT / "docs" / "ui" / "FEATURE-MAP.md").read_text(encoding="utf-8")
    parser = build_parser()
    for action in parser._actions:
        if action.dest == "help":
            continue
        for option in action.option_strings:
            assert option in text
    assert "needs decision" not in text
    assert "CLI only, unchanged" in text
    assert "not run before either, unchanged" in text
    assert "`upscale` subcommand" in text


def test_shell_slots_replace_the_second_toolbar() -> None:
    _app()
    set_language("en")
    window = SimpleWindow()
    source = (_ROOT / "enlarge" / "ui" / "simple_window.py").read_text(encoding="utf-8")
    assert "gfpgan" not in source
    assert "seedvr" not in source
    assert window._settings.get("enlarge.face") is None
    assert window._settings.get("enlarge.generative") is None
    assert window.simple.get("enlarge.long_side") == 0
    assert window.findChild(QPushButton, "actionMore") is None
    assert window.findChild(QPushButton, "overflow") is not None
    compare = window.view.root.findChild(QPushButton, "moreLike")
    again = window.view.root.findChild(QPushButton, "usePrompt")
    save = window.view.root.findChild(QPushButton, "savePicture")
    folder = window.view.root.findChild(QPushButton, "openFolder")
    assert compare is not None and compare.text() == "Compare before/after"
    assert again is not None and again.text() == "Enlarge again"
    assert save is not None and folder is not None
    for name in ("morePreview", "moreGpu", "moreAbout", "moreHash", "moreWeights"):
        assert window.findChild(QAction, name) is not None
    window.view.set_expert(True)
    side = window.view.root.findChild(QSpinBox, "enlarge.long_side")
    assert side is not None
    assert side.specialValueText() == "Off"
    assert side.value() == 0
    window.view.set_prompt_text("picture.png")
    assert window.simple.generate()
    assert window.simple.finish("wipe-1") is not None
    window.view.refresh()
    app = _app()
    for _ in range(3):
        app.processEvents()
    window.show()
    for _ in range(3):
        app.processEvents()
    slider = window.view.root.findChild(QSlider, "compareSlider")
    assert slider is not None and not slider.isHidden()
    window.open_batch()
    batch = window.findChild(QDialog, "batchScreen")
    assert batch is not None
    for name in ("batchAdd", "batchFolder", "batchRemove", "batchClear", "batchRun", "batchCancel"):
        assert batch.findChild(QPushButton, name) is not None
    box = batch.findChild(QCheckBox, "batchSubfolders")
    assert box is not None and box.text() == "Include subfolders"
    window._open_compare()
    assert window.findChild(QDialog, "compareScreen") is not None
    window.close()


def test_long_side_zero_keeps_the_size_and_the_command_names_the_flags(tmp_path: Path) -> None:
    source = tmp_path / "picture.png"
    source.write_bytes(b"not-read")
    assert model_scale(source, "2", 0) == 2
    settings = JobSettings(
        scale_text="4",
        long_side=1920,
        image_format="avif",
        quality=60,
        lossless=False,
        model_id="model",
        tile=0,
        double=True,
        device="auto",
        custom_models="",
        beside=True,
        output_dir="",
        subfolders=True,
    )
    line = command_line(source, settings)
    assert "--long-side" in line
    assert "--double" in line
    assert "--dry-run" in line
    assert "--json-progress" not in line
    assert "--lock-timeout" not in line
