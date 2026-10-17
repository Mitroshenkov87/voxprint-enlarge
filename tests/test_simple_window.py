"""The shared simple screen: one settings state, registry ids, and the wipe."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QPushButton, QSlider

from enlarge.ipc.server import MainWindow
from enlarge.ui.simple_window import (
    ENLARGE_SURFACE,
    SimpleWindow,
    known_model_ids,
    preset_model_ids,
    registry_models,
)
from voxprint_common.i18n import set_language
from voxprint_common.ui.presets import PresetMatch
from voxprint_common.ui.simple_image import Phase
from voxprint_common.ui.variants import ResultKind


def _app() -> QApplication:
    running = QApplication.instance()
    if isinstance(running, QApplication):
        return running
    return QApplication([])


def test_presets_name_registry_ids_not_files() -> None:
    ids = {model.id for model in registry_models()}
    written = preset_model_ids()
    assert set(written) == {"photo", "drawing", "old"}
    assert set(written.values()) == known_model_ids()
    assert set(written.values()) <= ids
    for model_id in written.values():
        assert not model_id.endswith((".pth", ".onnx", ".pt"))
        assert "/" not in model_id


def test_preset_edit_reset_and_mode_keep_one_state() -> None:
    _app()
    set_language("en")
    window = SimpleWindow()
    assert window.simple.get("enlarge.model") == preset_model_ids()["photo"]
    assert window.simple.get("enlarge.scale") == "2"
    assert window.simple.get("enlarge.format") == "avif"
    before = window.simple.snapshot()
    window.view.set_expert(True)
    assert window.simple.snapshot() == before
    window.view.set_expert(False)
    assert window.simple.snapshot() == before
    window.view.set_expert(True)
    window.simple.set_value("enlarge.tile", "512")
    status = window.simple.preset_status("look")
    assert status is not None
    assert status.kind is PresetMatch.MODIFIED
    assert status.option_id == "photo"
    window.view.refresh()
    chip = window.view.root.findChild(QPushButton, "chip-look-photo")
    assert chip is not None
    assert chip.text() == "Photo + changes"
    reset = window.view.root.findChild(QPushButton, "expertReset")
    assert reset is not None and reset.text() == "Reset to Photo"
    window.simple.reset_group("look")
    assert window.simple.get("enlarge.model") == preset_model_ids()["photo"]
    assert window.simple.get("enlarge.tile") == "256"
    restored = window.simple.preset_status("look")
    assert restored is not None and restored.kind is PresetMatch.EXACT
    window.close()


def test_compare_wipe_is_the_result_and_the_old_window_is_not_launched() -> None:
    _app()
    assert MainWindow is SimpleWindow
    assert ENLARGE_SURFACE.result_kind is ResultKind.COMPARE
    window = SimpleWindow()
    window.view.set_prompt_text("picture.png")
    assert window.simple.generate()
    assert window.simple.phase is Phase.BUSY
    assert window.simple.finish("result-1") is not None
    window.view.refresh()
    slider = window.view.root.findChild(QSlider, "compareSlider")
    assert slider is not None
    assert slider.value() == 500
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
        "en-wipe.png": ("en", False, True),
        "ru-simple.png": ("ru", False, False),
        "ru-expert.png": ("ru", True, False),
        "ru-wipe.png": ("ru", False, True),
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
