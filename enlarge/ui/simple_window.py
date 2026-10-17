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

from enlarge import suite
from enlarge.appinfo import APP_DISPLAY_NAME, resource_dir
from enlarge.images import is_image, plan_outputs
from enlarge.ipc.capabilities import gpu_capability
from enlarge.ui.jobs import BatchThread
from enlarge.ui.models import start_manage_models
from enlarge.ui.preset_map import load_enlarge_presets, picture_map
from voxprint_common.i18n import add_catalogs, tr
from voxprint_common.mdl.catalog import Model, load_manifest
from voxprint_common.paths import state_dir
from voxprint_common.ui.expert import ExpertMode
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
from voxprint_engine.registry import load_builtin
from voxprint_engine.worker import UpscaleRequest
from voxprint_engine.worker.lanczos import output_scale

# The registered enlarge surface is a still picture with no wipe, and its
# action bar adds picture-generation commands. The plan's file list, batch
# result, and compare slot are not on that surface yet. Clarity's compare
# result is the wipe. Video's action bar is Save and Open folder. See
# docs/SIMPLE-LAYOUT-GAPS.md.
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

_GPU_VALUES = {"auto": "enlarge.gpu.auto", "cuda:0": "enlarge.gpu.cuda0", "cuda:1": "enlarge.gpu.cuda1"}
_TILES = ("auto", "128", "256", "512", "768")


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
    photographs = picture_map()[0].model_id
    default_model = photographs if any(model.id == photographs for model in models) else models[0].id
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
                choices=tuple(Choice(model.id, model.id) for model in models),
            ),
            SettingSpec(
                "enlarge.scale",
                "enlarge.scale",
                "enlarge.scale.help",
                SettingType.ENUM,
                "4",
                expert=True,
                group="enlarge",
                choices=(Choice("2", "enlarge.scale.2"), Choice("4", "enlarge.scale.4")),
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
                "enlarge.lossless",
                "enlarge.lossless",
                "enlarge.lossless.help",
                SettingType.BOOL,
                False,
                expert=True,
                group="enlarge",
            ),
            SettingSpec(
                "enlarge.tile",
                "enlarge.tile",
                "enlarge.tile.help",
                SettingType.ENUM,
                "auto",
                expert=True,
                group="enlarge",
                choices=tuple(Choice(value, f"enlarge.tile.{value}") for value in _TILES),
            ),
            SettingSpec(
                "enlarge.double",
                "enlarge.double",
                "enlarge.double.help",
                SettingType.BOOL,
                False,
                expert=True,
                group="enlarge",
            ),
            SettingSpec(
                "enlarge.factor",
                "enlarge.factor",
                "enlarge.factor.help",
                SettingType.TEXT,
                "4×",
                expert=True,
                group="enlarge",
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
            SettingSpec(
                "enlarge.beside",
                "enlarge.beside",
                "enlarge.beside.help",
                SettingType.BOOL,
                True,
                expert=True,
                group="enlarge",
            ),
            SettingSpec(
                "enlarge.output_dir",
                "enlarge.output_dir",
                "enlarge.output_dir.help",
                SettingType.PATH,
                "",
                expert=True,
                group="enlarge",
            ),
            SettingSpec(
                "enlarge.custom_models",
                "enlarge.custom_models",
                "enlarge.custom_models.help",
                SettingType.PATH,
                "",
                expert=True,
                group="enlarge",
            ),
            SettingSpec(
                "enlarge.shared_models",
                "enlarge.shared_models",
                "enlarge.shared_models.help",
                SettingType.PATH,
                "",
                expert=True,
                group="enlarge",
            ),
            SettingSpec(
                "enlarge.licence",
                "enlarge.licence",
                "enlarge.licence.help",
                SettingType.TEXT,
                "",
                expert=True,
                group="enlarge",
            ),
            SettingSpec(
                "enlarge.subfolders",
                "enlarge.subfolders",
                "enlarge.subfolders.help",
                SettingType.BOOL,
                True,
                expert=False,
                group="enlarge",
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
        self._syncing = False
        self.simple = SimpleLayout(
            presets=load_enlarge_presets(self._settings),
            settings=self._settings,
            state_dir=state_dir() / "enlarge",
            gate=_Allow(),
            state_name="image-settings.json",
            history_name="image-history.json",
        )
        self._align_from_suite()
        self._pull_derived()
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
        warning = _gpu_warning()
        if warning:
            self.view.set_note(warning)

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

    def _align_from_suite(self) -> None:
        current = suite.gpu()
        if current in _GPU_VALUES and self.simple.get("enlarge.gpu") != current:
            self.simple.set_value("enlarge.gpu", current)
        shared = _shared_text()
        if self.simple.get("enlarge.shared_models") != shared:
            self.simple.set_value("enlarge.shared_models", shared)

    def _pull_derived(self) -> bool:
        """Write the licence line and the resulting factor. Return True when either changed."""
        changed = False
        licence = _licence_line(str(self.simple.get("enlarge.model")))
        if self.simple.get("enlarge.licence") != licence:
            self.simple.set_value("enlarge.licence", licence)
            changed = True
        factor = _factor_text(self.simple.get("enlarge.scale"), self.simple.get("enlarge.double"))
        if self.simple.get("enlarge.factor") != factor:
            self.simple.set_value("enlarge.factor", factor)
            changed = True
        return changed

    def _push_suite(self) -> None:
        choice = str(self.simple.get("enlarge.gpu"))
        if choice in _GPU_VALUES and choice != suite.gpu():
            suite.set_gpu(choice)
        chosen = str(self.simple.get("enlarge.shared_models") or "").strip()
        stored = suite.models_dir()
        shown = "" if stored is None else str(stored)
        baseline = shown or str(suite.resolved_models_dir())
        if chosen == baseline:
            return
        suite.set_models_dir(Path(chosen) if chosen else None)

    def _after_paint(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._push_suite()
            changed = self._pull_derived()
            if self._job is not None and self.simple.phase is not Phase.BUSY:
                self._job.cancel()
            if changed:
                self.view.refresh()
        finally:
            self._syncing = False

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
        image_format = str(self.simple.get("enlarge.format"))
        lossless = bool(self.simple.get("enlarge.lossless"))
        if lossless and image_format != "jxl":
            self.simple.cancel()
            self.view.set_note(tr("enlarge.error.lossless"))
            self.view.refresh()
            return
        self._push_suite()
        scale = int(str(self.simple.get("enlarge.scale")))
        tile_text = str(self.simple.get("enlarge.tile"))
        quality = self.simple.get("enlarge.quality")
        if isinstance(quality, bool) or not isinstance(quality, int):
            quality = DEFAULT_QUALITY
        double = bool(self.simple.get("enlarge.double"))
        folder_text = str(self.simple.get("enlarge.output_dir") or "").strip()
        beside = bool(self.simple.get("enlarge.beside"))
        folder = None if beside or not folder_text else Path(folder_text)
        destination = plan_outputs([source], folder, scale, image_format, double)[0]
        job_id = uuid.uuid4().hex
        request = UpscaleRequest(
            id=job_id,
            src=str(source),
            dst=str(destination),
            scale=scale,
            format=image_format,
            quality=quality,
            lossless=lossless,
            model_id=str(self.simple.get("enlarge.model")),
            tile=0 if tile_text == "auto" else int(tile_text),
            double=double,
            device=suite.gpu(),
            custom_models_dir=str(self.simple.get("enlarge.custom_models") or ""),
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
    return {row.option_id: row.model_id for row in picture_map()}


def known_model_ids() -> frozenset[str]:
    """Return the placeholder registry ids the picture presets name."""
    return frozenset(preset_model_ids().values())


def _shared_text() -> str:
    stored = suite.models_dir()
    if stored is not None:
        return str(stored)
    return str(suite.resolved_models_dir())


def _factor_text(scale: object, double: object) -> str:
    try:
        factor = output_scale(int(str(scale)), bool(double))
    except (TypeError, ValueError):
        return ""
    return f"{factor}×"


def _licence_line(model_id: str) -> str:
    model = load_builtin(release=False).get(model_id)
    if model is None:
        return ""
    licence = model.licence.strip()
    author = model.author.strip()
    if author and licence:
        return f"{author} · {licence}"
    return licence or author


def _gpu_warning() -> str:
    try:
        report = gpu_capability()
    except (OSError, RuntimeError, ValueError):
        return ""
    if report.get("status") == "warn":
        return tr("enlarge.gpu.warn")
    return ""
