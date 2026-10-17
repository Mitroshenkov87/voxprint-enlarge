"""The main window on the shared simple screen.

The old window stays in :mod:`enlarge.ui.window` and is no longer opened.
This module configures :class:`voxprint_common.ui.simple_image.SimpleLayout`.
It does not replace that screen.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QWidget

from enlarge.appinfo import APP_DISPLAY_NAME, resource_dir
from enlarge.images import is_image, plan_outputs
from enlarge.ui.jobs import BatchThread
from enlarge.ui.models import start_manage_models
from enlarge import suite
from voxprint_common.i18n import add_catalogs, tr
from voxprint_common.mdl.catalog import Model, load_manifest
from voxprint_common.paths import state_dir
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.presets import load_presets
from voxprint_common.ui.settings import Choice, SettingSpec, SettingType, Settings
from voxprint_common.ui.simple_image import Phase, SimpleLayout, SimpleWidget
from voxprint_common.ui.theme import build_shell_style
from voxprint_common.ui.variants import (
    WINDOW_HEIGHT,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_WIDTH,
    AppSurface,
    InputKind,
    ResultKind,
    VariantId,
)
from voxprint_engine.formats import DEFAULT_FORMAT, DEFAULT_QUALITY, choices
from voxprint_engine.worker import UpscaleRequest

# The registered enlarge surface is a still picture with no wipe, and its
# action bar adds picture-generation commands. Clarity's compare result is the
# wipe. Video's action bar is Save and Open folder, which is the closest bar
# the shell offers for a file in and a picture out. See docs/SIMPLE-LAYOUT-GAPS.md.
ENLARGE_SURFACE = AppSurface(
    "enlarge",
    VariantId.VIDEO,
    InputKind.FILE,
    ResultKind.COMPARE,
    "simple.action.enlarge",
    "enlarge.empty",
    "simple.drop.picture",
    "enlarge",
    "simple.drop.picture",
)

_PHOTO = "4x-nomos2-realplksr-dysample"
_DRAWING = "realesrgan-x4plus-anime6b"
_OLD = "4x-nomos2-hq-dat2"
_GPU_VALUES = {"auto": "enlarge.gpu.auto", "cuda:0": "enlarge.gpu.cuda0", "cuda:1": "enlarge.gpu.cuda1"}


class _Allow:
    """Let a picture start and stay. Enlarge has no prompt to review."""

    def review_prompt(self, prompt: str) -> str | None:
        """Allow every file name."""
        del prompt
        return None

    def review_result(self, prompt: str, item_id: str) -> str | None:
        """Allow every finished picture."""
        del prompt, item_id
        return None


def registry_models(manifest: Path | None = None) -> tuple[Model, ...]:
    """Return Enlarge's models from the MDL manifest, in file order.

    Args:
        manifest: ``mdl.json``, or the copy beside the application.

    Raises:
        ValueError: The manifest has no model id.
    """
    path = resource_dir() / "mdl.json" if manifest is None else manifest
    models = tuple(model for model in load_manifest(path).models if model.id and not model.unavailable)
    if not models:
        raise ValueError(f"{path} has no models")
    return models


def enlarge_settings(models: tuple[Model, ...]) -> Settings:
    """Return the one settings list the chips and the expert panel share.

    Args:
        models: Registry models. Their ids are the model choices.
    """
    model_choices = tuple(Choice(model.id, f"enlarge.model.{model.id}") for model in models)
    default_model = _PHOTO if any(model.id == _PHOTO for model in models) else models[0].id
    format_choices = tuple(Choice(key, f"enlarge.format.{key}") for _label, key in choices())
    if any(item.value == DEFAULT_FORMAT for item in format_choices):
        default_format = DEFAULT_FORMAT
    else:
        default_format = format_choices[0].value
    return Settings(
        (
            SettingSpec(
                "enlarge.model",
                "enlarge.model",
                "enlarge.model.help",
                SettingType.ENUM,
                default_model,
                expert=True,
                group="enlarge",
                choices=model_choices,
            ),
            SettingSpec(
                "enlarge.scale",
                "enlarge.scale",
                "enlarge.scale.help",
                SettingType.ENUM,
                "2",
                expert=True,
                group="enlarge",
                choices=(Choice("2", "enlarge.scale.2"), Choice("4", "enlarge.scale.4")),
            ),
            SettingSpec(
                "enlarge.tile",
                "enlarge.tile",
                "enlarge.tile.help",
                SettingType.ENUM,
                "auto",
                expert=True,
                group="enlarge",
                choices=tuple(
                    Choice(value, f"enlarge.tile.{value}") for value in ("auto", "128", "256", "512", "768")
                ),
            ),
            SettingSpec(
                "enlarge.face",
                "enlarge.face",
                "enlarge.face.help",
                SettingType.BOOL,
                False,
                expert=True,
                group="enlarge",
            ),
            SettingSpec(
                "enlarge.denoise",
                "enlarge.denoise",
                "enlarge.denoise.help",
                SettingType.FLOAT,
                0.0,
                expert=True,
                group="enlarge",
                minimum=0.0,
                maximum=1.0,
            ),
            SettingSpec(
                "enlarge.format",
                "enlarge.format",
                "enlarge.format.help",
                SettingType.ENUM,
                default_format,
                expert=True,
                group="enlarge",
                choices=format_choices,
            ),
            SettingSpec(
                "enlarge.quality",
                "enlarge.quality",
                "enlarge.quality.help",
                SettingType.INT,
                DEFAULT_QUALITY,
                expert=True,
                group="enlarge",
                minimum=1,
                maximum=100,
            ),
            SettingSpec(
                "enlarge.gpu",
                "enlarge.gpu",
                "enlarge.gpu.help",
                SettingType.ENUM,
                "auto",
                expert=True,
                group="enlarge",
                choices=tuple(Choice(value, key) for value, key in _GPU_VALUES.items()),
            ),
        )
    )


class SimpleWindow(QWidget):
    """Enlarge on the shared simple screen. Simple mode is the default."""

    def __init__(self) -> None:
        """Build the shared screen and restore the suite's Expert mode flag."""
        super().__init__()
        add_catalogs(Path(__file__).resolve().parent / "locales")
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setObjectName("shell")
        self.setStyleSheet(build_shell_style(False))
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self._models = registry_models()
        self._settings = enlarge_settings(self._models)
        self._results: dict[str, Path] = {}
        self._job: BatchThread | None = None
        self._manage: object | None = None
        presets = load_presets(Path(__file__).resolve().parent / "presets" / "enlarge.json", self._settings)
        self.simple = SimpleLayout(
            presets=presets,
            settings=self._settings,
            state_dir=state_dir() / "enlarge",
            gate=_Allow(),
            state_name="image-settings.json",
            history_name="image-history.json",
        )
        self._align_gpu()
        self._expert = ExpertMode()
        self.view = SimpleWidget(
            self.simple,
            self._settings,
            on_generate=self._start,
            on_command=self._command,
            on_refresh=self._after_paint,
            surface=ENLARGE_SURFACE,
        )
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.view.root)
        self._stop_expert = self._expert.subscribe(self.view.set_expert)
        self.view.set_expert(self._expert.enabled())

    def add_paths(self, paths: list[Path]) -> None:
        """Put the first path in the drop area.

        Args:
            paths: Files another app asked this window to show.
        """
        if paths:
            self.view.set_prompt_text(str(paths[0]))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's name
        """Stop listening for Expert mode when the window closes."""
        self._stop_expert()
        super().closeEvent(event)

    def _align_gpu(self) -> None:
        current = suite.gpu()
        if current in _GPU_VALUES and self.simple.get("enlarge.gpu") != current:
            self.simple.set_value("enlarge.gpu", current)

    def _after_paint(self) -> None:
        choice = str(self.simple.get("enlarge.gpu"))
        if choice in _GPU_VALUES and choice != suite.gpu():
            suite.set_gpu(choice)
        if self._job is not None and self.simple.phase is not Phase.BUSY:
            self._job.cancel()

    def _command(self, command_id: str) -> None:
        if command_id == "view.toggle_expert_mode":
            self._expert.toggle()
            return
        if command_id == "models.manage":
            self._manage = start_manage_models()
            return
        if command_id == "file.save":
            self._save()
            return
        if command_id == "file.open_folder":
            self._open_folder()
            return
        if command_id == "file.open":
            self._open()

    def _open(self) -> None:
        selected, _used = QFileDialog.getOpenFileName(self, APP_DISPLAY_NAME)
        if selected:
            self.view.set_prompt_text(selected)

    def _open_folder(self) -> None:
        target = self._current_result()
        source = Path(self.simple.prompt)
        folder = target.parent if target is not None else source.parent
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _save(self) -> None:
        source = self._current_result()
        if source is None or not source.is_file():
            self.view.set_note(tr("enlarge.error.missing"))
            return
        selected, _used = QFileDialog.getSaveFileName(self, APP_DISPLAY_NAME, str(source))
        if not selected:
            return
        destination = Path(selected)
        if destination.resolve() != source.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        self.view.set_note(tr("enlarge.note.saved"))

    def _current_result(self) -> Path | None:
        item_id = self.simple.selected
        if item_id is None:
            return None
        return self._results.get(item_id)

    def _start(self) -> None:
        source = Path(self.simple.prompt)
        if not is_image(source):
            self.simple.cancel()
            self.view.set_note(tr("enlarge.error.missing"))
            self.view.refresh()
            return
        scale = int(str(self.simple.get("enlarge.scale")))
        image_format = str(self.simple.get("enlarge.format"))
        tile_text = str(self.simple.get("enlarge.tile"))
        quality = self.simple.get("enlarge.quality")
        if isinstance(quality, bool) or not isinstance(quality, int):
            quality = DEFAULT_QUALITY
        destination = plan_outputs([source], None, scale, image_format, False)[0]
        job_id = uuid.uuid4().hex
        request = UpscaleRequest(
            id=job_id,
            src=str(source),
            dst=str(destination),
            scale=scale,
            format=image_format,
            quality=quality,
            model_id=str(self.simple.get("enlarge.model")),
            tile=0 if tile_text == "auto" else int(tile_text),
            device=suite.gpu(),
            app_name=APP_DISPLAY_NAME,
        )
        self._job = BatchThread([request], lambda: self.simple.phase is not Phase.BUSY)
        self._job.progressed.connect(self._on_progress)
        self._job.item_finished.connect(self._on_item)
        self._job.start()

    def _on_progress(self, _job_id: str, fraction: float, message: str) -> None:
        if self.simple.phase is not Phase.BUSY:
            return
        bounded = min(1.0, max(0.0, float(fraction)))
        label = message.strip() or tr("simple.progress")
        self.simple.progress(int(round(bounded * 100)), 100, label)
        self.view.refresh()

    def _on_item(self, job_id: str, ok: bool, message: str, dst: str) -> None:
        if self.simple.phase is not Phase.BUSY:
            return
        if ok and dst:
            self._results[job_id] = Path(dst)
            self.simple.finish(job_id)
        else:
            self.simple.block(message.strip() or tr("enlarge.error.failed"))
        self.view.refresh()


def preset_model_ids() -> dict[str, str]:
    """Return each picture preset and the registry id it writes."""
    settings = enlarge_settings(registry_models())
    document = load_presets(Path(__file__).resolve().parent / "presets" / "enlarge.json", settings)
    group = document.group("look")
    if group is None:
        return {}
    return {option.option_id: str(option.as_map()["enlarge.model"]) for option in group.options}


def known_model_ids() -> frozenset[str]:
    """Return the registry ids the picture presets are allowed to name."""
    return frozenset({_PHOTO, _DRAWING, _OLD})
