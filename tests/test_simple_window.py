"""The shared simple screen: one settings state, registry ids, and the wipe."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QColor, QFontMetrics, QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QWidget,
)

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
    shown_label,
)
from voxprint_common import i18n
from voxprint_common.i18n import add_catalogs, set_language
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
    assert window.simple.get("enlarge.long_side") == 0
    assert window.simple.get("enlarge.beside") is True
    assert window.simple.get("enlarge.scope") == "one"
    assert window.simple.get("enlarge.lossless") is False
    window.simple.choose("quality", "best")
    assert window.simple.get("enlarge.quality") == 95
    window.simple.choose("quality", "smaller")
    assert window.simple.get("enlarge.quality") == 50
    window.simple.choose("quality", "balanced")
    window.simple.choose("scale", "4k")
    assert window.simple.get("enlarge.scale") == "4"
    assert window.simple.get("enlarge.long_side") == 3840
    window.simple.choose("scale", "8k")
    assert window.simple.get("enlarge.long_side") == 7680
    window.simple.choose("scale", "2x")
    assert window.simple.get("enlarge.scale") == "2"
    assert window.simple.get("enlarge.long_side") == 0
    window.simple.choose("scale", "4x")
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
    window.simple.set_value("enlarge.long_side", 1920)
    status = window.simple.preset_status("scale")
    assert status is not None
    assert status.kind is PresetMatch.MODIFIED
    assert status.option_id == "4x"
    window.resize(1200, 900)
    window.show()
    for _ in range(4):
        _app().processEvents()
    window.view.refresh()
    chip = window.view.root.findChild(QPushButton, "chip-scale-4x")
    assert chip is not None
    assert chip.text() == "Four times as big + changes"
    side = window.view.root.findChild(QSpinBox, "enlarge.long_side")
    assert side is not None and side.value() == 1920
    reset = window.view.root.findChild(QPushButton, "expertReset")
    assert reset is not None and reset.text() == "Reset to Four times as big"
    window.simple.reset_group("scale")
    assert window.simple.get("enlarge.long_side") == 0
    restored = window.simple.preset_status("scale")
    assert restored is not None and restored.kind is PresetMatch.EXACT
    window.simple.set_value("enlarge.quality", 80)
    quality = window.simple.preset_status("quality")
    assert quality is not None and quality.kind is PresetMatch.CUSTOM
    window.simple.choose("quality", "balanced")
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


_CHIP_ROWS = ("pictures", "scale", "quality", "filetype", "save", "scope")


def test_every_chip_row_renders_in_en_ru_and_de() -> None:
    catalogs = {
        lang: json.loads(
            (_ROOT / "enlarge" / "ui" / "locales" / f"{lang}.json").read_text(encoding="utf-8")
        )
        for lang in ("en", "ru", "de")
    }
    russian = catalogs["ru"]
    assert catalogs["en"]["enlarge.preset.pictures"] == "What's in the picture"
    assert russian["enlarge.preset.pictures"] == "Что на картинке"
    assert catalogs["de"]["enlarge.preset.pictures"] == "Was auf dem Bild"
    assert russian["enlarge.preset.quality"] == "Качество"
    assert russian["enlarge.preset.quality.smaller"] == "Файл поменьше"
    assert russian["enlarge.preset.quality.balanced"] == "Сбалансировано"
    assert russian["enlarge.preset.quality.best"] == "Лучшее качество"
    assert russian["enlarge.preset.save"] == "Куда сохранить"
    assert russian["enlarge.preset.save.beside"] == "Рядом с изображением"
    assert russian["enlarge.preset.save.folder"] == "Выбрать папку…"
    assert russian["enlarge.preset.scope"] == "Что увеличить"
    assert russian["enlarge.preset.scope.one"] == "Одно изображение"
    assert russian["enlarge.preset.scope.folder"] == "Целую папку"
    assert russian["enlarge.preset.scale.4k"] == "До 4K"
    assert russian["enlarge.preset.scale.8k"] == "До 8K"
    app = _app()
    opened: list[SimpleWindow] = []
    try:
        for lang in ("en", "ru", "de"):
            set_language(lang)
            window = SimpleWindow()
            opened.append(window)
            window.resize(1200, 900)
            window.show()
            for _ in range(4):
                app.processEvents()
            catalog = catalogs[lang]
            assert tuple(row.row_id for row in window.simple.rows) == _CHIP_ROWS
            for row in window.simple.rows:
                label = window.findChild(QLabel, f"preset-{row.row_id}")
                assert label is not None and label.isVisible()
                assert label.text() == catalog[row.label_key]
                assert label.text().strip() and label.text() != row.label_key
                for choice in row.choices:
                    safe = "".join(char if char.isalnum() else "-" for char in choice.value)
                    button = window.findChild(QPushButton, f"chip-{row.row_id}-{safe}")
                    assert button is not None and button.isVisible(), choice.label_key
                    assert button.text().strip()
                    assert button.text() != choice.label_key
                    assert catalog[choice.label_key] in button.text()
    finally:
        set_language("en")
        for window in opened:
            window.close()


def test_blank_translation_falls_back_to_english_and_stays_visible(tmp_path: Path) -> None:
    folder = tmp_path / "locales"
    folder.mkdir()
    (folder / "ru.json").write_text(
        json.dumps(
            {
                "enlarge.preset.quality": "",
                "enlarge.preset.quality.balanced": "   ",
                "enlarge.preset.save": "",
            }
        ),
        encoding="utf-8",
    )
    _app()
    set_language("ru")
    window = SimpleWindow()
    add_catalogs(folder)
    app = _app()
    window.resize(1200, 900)
    window.show()
    try:
        window.view.refresh()
        for _ in range(4):
            app.processEvents()
        quality = window.findChild(QLabel, "preset-quality")
        balanced = window.findChild(QPushButton, "chip-quality-balanced")
        save = window.findChild(QLabel, "preset-save")
        pictures = window.findChild(QLabel, "preset-pictures")
        assert quality is not None and quality.isVisible() and quality.text() == "Quality"
        assert balanced is not None and balanced.isVisible() and balanced.text() == "Balanced"
        assert save is not None and save.isVisible() and save.text() == "Save to"
        assert pictures is not None and pictures.isVisible() and pictures.text() == "Что на картинке"
        assert shown_label("enlarge.preset.quality") == "Quality"
    finally:
        if folder in i18n._extra:
            i18n._extra.remove(folder)
        i18n._catalogs.clear()
        set_language("en")
        window.close()


def test_history_thumb_uses_the_result_image(tmp_path: Path) -> None:
    _app()
    set_language("ru")
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    _solid(before, QColor(180, 30, 30))
    _solid(after, QColor(30, 30, 180))
    window = SimpleWindow()
    window.resize(1200, 900)
    window.show()
    window.view.set_prompt_text(str(before))
    assert window.simple.generate()
    assert window.simple.finish("same-image") is not None
    window.remember("same-image", after)
    window.view.refresh()
    for _ in range(4):
        _app().processEvents()
    button = window.findChild(QPushButton, "thumb-same-image")
    assert button is not None and button.isVisible()
    image = button.icon().pixmap(QSize(80, 80)).toImage()
    assert not image.isNull()
    left = image.pixelColor(12, 40)
    right = image.pixelColor(68, 40)
    assert left.red() > 140 and left.blue() < 80
    assert right.blue() > 140 and right.red() < 80
    window.close()
    set_language("en")


def test_drop_history_and_expert_combos_stay_readable(tmp_path: Path) -> None:
    app = _app()
    set_language("en")
    window = SimpleWindow()
    window.resize(1200, 900)
    window.show()
    for _ in range(4):
        app.processEvents()
    strip = window.findChild(QScrollArea, "historyStrip")
    drop = window.findChild(QLabel, "fileDrop")
    assert strip is not None and strip.isHidden()
    assert drop is not None and drop.text() == "Drop a picture or click to choose"
    chosen = tmp_path / "shots" / "before.png"
    chosen.parent.mkdir()
    chosen.write_bytes(b"png")
    window.view.set_prompt_text(str(chosen))
    assert drop.text() == "before.png · replace"
    assert "Drop a picture" not in drop.text()
    assert "shots" not in drop.text()
    window.view.set_expert(True)
    for _ in range(6):
        app.processEvents()
    window._dress_expert()
    for _ in range(4):
        app.processEvents()
    seen = False
    for combo in window.view.root.findChildren(QComboBox):
        if not combo.isVisible() or combo.count() == 0:
            continue
        seen = True
        text = combo.currentText()
        advance = QFontMetrics(combo.font()).horizontalAdvance(text)
        assert "…" not in text
        assert combo.width() >= advance + 28, (combo.objectName(), text, combo.width(), advance)
    assert seen
    pictures = window.findChild(QLabel, "preset-pictures")
    assert pictures is not None and pictures.text() == "What's in the picture"
    window.view.set_expert(False)
    assert window.simple.generate()
    assert window.simple.finish("hist-1") is not None
    window.view.refresh()
    for _ in range(4):
        app.processEvents()
    assert not strip.isHidden()
    assert strip.height() >= 80
    window.close()

    set_language("ru")
    russian = SimpleWindow()
    russian.view.set_prompt_text(str(chosen))
    ru_drop = russian.findChild(QLabel, "fileDrop")
    ru_pictures = russian.findChild(QLabel, "preset-pictures")
    assert ru_drop is not None and ru_drop.text() == "before.png · заменить"
    assert ru_pictures is not None and ru_pictures.text() == "Что на картинке"
    russian.close()

    set_language("de")
    german = SimpleWindow()
    german.view.set_prompt_text(str(chosen))
    de_drop = german.findChild(QLabel, "fileDrop")
    de_pictures = german.findChild(QLabel, "preset-pictures")
    assert de_drop is not None and de_drop.text() == "before.png · ersetzen"
    assert de_pictures is not None and de_pictures.text() == "Was auf dem Bild"
    german.close()
    set_language("en")


def _solid(path: Path, color: QColor) -> None:
    image = QImage(48, 32, QImage.Format.Format_RGB32)
    image.fill(color)
    assert image.save(str(path))


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
    assert "interim, until a shell command id" in text
    assert "needs worker support" in text
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
    for name in ("moreCredits", "moreAbout", "morePreview", "moreGpu"):
        assert window.findChild(QAction, name) is not None
    assert window.findChild(QAction, "moreFolder") is None
    assert window.findChild(QAction, "moreHash") is None
    assert window.findChild(QAction, "moreWeights") is None
    credits = window.findChild(QAction, "moreCredits")
    preview = window.findChild(QAction, "morePreview")
    assert credits is not None and credits.text() == "Credits"
    assert preview is not None and preview.text() == "Preview file names"
    whole = window.findChild(QPushButton, "chip-scope-folder")
    assert whole is not None and whole.text() == "A whole folder"
    whole.click()
    assert window.findChild(QDialog, "batchScreen") is not None
    for chip, label in (
        ("chip-pictures-photographs", "Photographs"),
        ("chip-pictures-detail", "Photographs, highest detail"),
        ("chip-pictures-drawings", "Drawings"),
        ("chip-scale-2x", "Twice as big"),
        ("chip-scale-4x", "Four times as big"),
        ("chip-scale-4k", "Up to 4K"),
        ("chip-scale-8k", "Up to 8K"),
        ("chip-quality-smaller", "Smaller file"),
        ("chip-quality-balanced", "Balanced"),
        ("chip-quality-best", "Best quality"),
        ("chip-save-beside", "Next to the picture"),
        ("chip-save-folder", "Choose a folder…"),
        ("chip-scope-one", "One picture"),
    ):
        button = window.findChild(QPushButton, chip)
        assert button is not None and button.text() == label
    assert window.findChild(QPushButton, "chip-face-on") is None
    assert window.findChild(QPushButton, "chip-denoise-on") is None
    window.simple.choose("scope", "one")
    window.resize(1200, 900)
    window.show()
    app = _app()
    for _ in range(4):
        app.processEvents()
    bar = window.findChild(QWidget, "actionBar")
    assert bar is not None and bar.isHidden()
    window._open_models()
    models = window.findChild(QDialog, "modelsScreen")
    assert models is not None
    assert models.findChild(QPushButton, "modelsHash") is not None
    assert models.findChild(QPushButton, "modelsRemove") is not None
    assert "StandardButton.No" in (_ROOT / "enlarge" / "ui" / "screens.py").read_text(encoding="utf-8")
    drop = window.findChild(QLabel, "fileDrop")
    assert drop is not None and drop.text().splitlines()[0] == "Drop a picture or click to choose"
    window.view.set_expert(True)
    app = _app()
    for _ in range(3):
        app.processEvents()
    factor = window.findChild(QLabel, "enlarge.factorValue")
    assert factor is not None and factor.text() == "4×"
    assert factor.sizeHint().height() < 36
    form = window.findChild(QWidget, "expertForm")
    assert form is not None and form.layout() is not None
    assert form.layout().contentsMargins().right() >= 8
    window.view.set_expert(True)
    factor = window.findChild(QLabel, "enlarge.factorValue")
    assert factor is not None and factor.text() == "4×"
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
    assert compare is not None and not compare.isHidden()
    slider = window.view.root.findChild(QSlider, "compareSlider")
    assert slider is not None and not slider.isHidden()
    window.open_batch()
    batch = window.findChild(QDialog, "batchScreen")
    assert batch is not None
    for name in ("batchAdd", "batchFolder", "batchRemove", "batchClear", "batchRun", "batchCancel"):
        assert batch.findChild(QPushButton, name) is not None
    box = batch.findChild(QCheckBox, "batchSubfolders")
    assert box is not None and box.text() == "Include subfolders"
    placeholder = batch.findChild(QWidget, "batchCompare")
    assert placeholder is not None
    assert str(placeholder.placeholder).startswith("Enlarge to compare before and after")
    assert "upscale" not in str(placeholder.placeholder).lower()
    set_language("ru")
    russian = SimpleWindow()
    ru_drop = russian.findChild(QLabel, "fileDrop")
    assert ru_drop is not None
    assert ru_drop.text().splitlines()[0] == "Перетащите изображение или нажмите, чтобы выбрать"
    hint = russian.findChild(QLabel, "emptyHint")
    assert hint is not None and hint.text() == "Здесь появится изображение"
    russian.resize(1200, 900)
    russian.show()
    russian.view.set_expert(True)
    for _ in range(6):
        app.processEvents()
    ru_drop = russian.findChild(QLabel, "fileDrop")
    assert ru_drop is not None
    assert ru_drop.text().splitlines()[0] == "Перетащите изображение или нажмите, чтобы выбрать"
    assert "картин" not in ru_drop.text()
    ru_form = russian.findChild(QWidget, "expertForm")
    assert ru_form is not None
    boxes = []
    for label in ru_form.findChildren(QLabel):
        if not label.isVisible() or not label.text().strip() or label.height() < 8:
            continue
        origin = label.mapTo(ru_form, label.rect().topLeft())
        boxes.append((label.text(), origin.x(), origin.y(), origin.x() + label.width(), origin.y() + label.height()))
    for index, earlier in enumerate(boxes):
        for later in boxes[index + 1 :]:
            apart = earlier[3] <= later[1] or later[3] <= earlier[1] or earlier[4] <= later[2] or later[4] <= earlier[2]
            assert apart, (earlier, later)
    ru_side = russian.findChild(QSpinBox, "enlarge.long_side")
    ru_format = russian.findChild(QWidget, "enlarge.format")
    assert ru_side is not None and ru_format is not None
    assert ru_side.width() >= ru_format.width() - 4
    factor_ru = russian.findChild(QLabel, "enlarge.factorValue")
    assert factor_ru is not None and factor_ru.text() == "4×" and factor_ru.height() >= 8
    russian.open_batch()
    ru_batch = russian.findChild(QDialog, "batchScreen")
    ru_compare = None if ru_batch is None else ru_batch.findChild(QWidget, "batchCompare")
    assert ru_compare is not None
    assert str(ru_compare.placeholder).startswith("Увеличьте, чтобы сравнить до и после")
    russian.close()
    set_language("en")
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
