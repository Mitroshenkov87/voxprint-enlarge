"""The shared simple screen: one settings state, registry ids, and the wipe."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QPushButton, QSlider

from enlarge.ipc.capabilities import ART, PHOTO_FAST, PHOTO_MAX
from enlarge.ipc.server import MainWindow
from enlarge.ui.preset_map import map_path, picture_map
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
    assert window.simple.get("enlarge.subfolders") is True
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
