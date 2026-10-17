"""The main window on the shared simple screen.

The old window stays in :mod:`enlarge.ui.window` and is no longer opened.
This module configures :class:`voxprint_common.ui.simple_image.SimpleLayout`.
It does not replace that screen.
"""
from __future__ import annotations

import contextlib
import io
import shutil
from collections.abc import Callable
from pathlib import Path

import main as entry

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QFileDialog, QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget

from enlarge import suite
from enlarge.appinfo import APP_DISPLAY_NAME, resource_dir, version_label
from enlarge.images import is_image
from enlarge.ipc.capabilities import gpu_capability
from enlarge.ui.compare import _load
from enlarge.ui.jobplan import JobSettings, as_int, fit_long_side, requests_for
from enlarge.ui.models import start_manage_models
from enlarge.ui.preset_map import load_enlarge_presets, picture_map
from enlarge.ui.runner import GuidedBatchThread
from enlarge.ui.screens import (
    BatchScreen,
    CompareScreen,
    JobHost,
    apply_suite_surface,
    show_credits,
    show_dry_run,
    show_hash,
    show_report,
    remove_weights,
)
from voxprint_common.i18n import add_catalogs, tr
from voxprint_common.mdl import ready
from voxprint_common.mdl.catalog import Model, load_manifest
from voxprint_common.paths import state_dir
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.settings import Choice, SettingSpec, SettingType, Settings
from voxprint_common.ui.simple_image import Phase, SimpleLayout, SimpleWidget
from voxprint_common.ui.variants import (
    WINDOW_HEIGHT,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_WIDTH,
    surface_for,
)
from voxprint_engine.formats import DEFAULT_FORMAT, DEFAULT_QUALITY, choices
from voxprint_engine.registry import load_builtin
from voxprint_engine.worker import UpscaleRequest
from voxprint_engine.worker.lanczos import output_scale

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
                "enlarge.long_side",
                "enlarge.long_side",
                "enlarge.long_side.help",
                SettingType.INT,
                0,
                expert=True,
                group="enlarge",
                minimum=0,
                maximum=20000,
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
                expert=False,
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
        )
    )


class SimpleWindow(QWidget, JobHost):
    """Enlarge on the shared simple screen. Simple mode is the default."""

    def __init__(self) -> None:
        """Build the shared screen and restore the suite's Expert mode flag."""
        super().__init__()
        add_catalogs(Path(__file__).resolve().parent / "locales")
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setObjectName("shell")
        apply_suite_surface(self)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self._models = registry_models()
        self._settings = enlarge_settings(self._models)
        self._results: dict[str, Path] = {}
        self._fits: dict[str, tuple[Path, int]] = {}
        self._include_subfolders = True
        self._wipe: QSlider | None = None
        self._result_label: QLabel | None = None
        self._job: GuidedBatchThread | None = None
        self._manage: object | None = None
        self._batch: BatchScreen | None = None
        self._compare: CompareScreen | None = None
        self._credits: QWidget | None = None
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
        self.surface = surface_for("enlarge")
        self.view = SimpleWidget(
            self.simple,
            self._settings,
            on_generate=self._start,
            on_command=self._command,
            on_refresh=self._after_paint,
            surface=self.surface,
        )
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.view.root)
        self._retarget_results()
        self._extend_overflow()
        self._install_full_drop()
        expert_impl = self.view.set_expert

        def watched(enabled: bool) -> None:
            expert_impl(enabled)
            self._dress_expert()
            self._fit_bold_controls()

        self.view.set_expert = watched  # type: ignore[method-assign]
        self._stop_expert = self._expert.subscribe(watched)
        watched(self._expert.enabled())
        warning = _gpu_warning()
        if warning:
            self.view.set_note(warning)

    def add_paths(self, paths: list[Path]) -> None:
        """Put the first path in the drop area.

        Args:
            paths: Files another app asked this window to show.
        """
        if len(paths) == 1 and paths[0].is_file():
            self.view.set_prompt_text(str(paths[0]))
            return
        if paths:
            self.open_batch(paths)

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
        factor = _factor_text(
            self.simple.get("enlarge.scale"),
            self.simple.get("enlarge.double"),
            self.simple.get("enlarge.long_side"),
        )
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
            view = getattr(self, "view", None)
            if changed and view is not None:
                view.refresh()
            if view is not None:
                self._dress_expert()
                self._fit_bold_controls()
                QTimer.singleShot(0, self._paint_wipe)
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
        if command_id == "view.settings":
            if not self._expert.enabled():
                self._expert.toggle()
            return
        if command_id == "file.open":
            self._open()

    def job_settings(self) -> JobSettings:
        """Current expert and simple values."""
        quality = self.simple.get("enlarge.quality")
        if isinstance(quality, bool) or not isinstance(quality, int):
            quality = DEFAULT_QUALITY
        tile_text = str(self.simple.get("enlarge.tile"))
        self._push_suite()
        return JobSettings(
            scale_text=str(self.simple.get("enlarge.scale")),
            long_side=as_int(self.simple.get("enlarge.long_side")),
            image_format=str(self.simple.get("enlarge.format")),
            quality=quality,
            lossless=bool(self.simple.get("enlarge.lossless")),
            model_id=str(self.simple.get("enlarge.model")),
            tile=0 if tile_text == "auto" else as_int(tile_text),
            double=bool(self.simple.get("enlarge.double")),
            device=suite.gpu(),
            custom_models=str(self.simple.get("enlarge.custom_models") or ""),
            beside=bool(self.simple.get("enlarge.beside")),
            output_dir=str(self.simple.get("enlarge.output_dir") or ""),
            subfolders=self._include_subfolders,
        )

    def models_ready(self) -> bool:
        """True when the shared models can run."""
        return _models_ready()

    def set_subfolders(self, enabled: bool) -> None:
        """Store the batch subfolders switch. It is not a simple-mode chip."""
        self._include_subfolders = enabled

    def remember(self, job_id: str, path: Path) -> None:
        """Keep a finished file so Compare and Save can find it."""
        self._results[job_id] = path

    def launch(
        self,
        requests: list[UpscaleRequest],
        cancel: Callable[[], bool],
        on_progress: Callable[[str, float, str], None],
        on_item: Callable[[str, bool, str, str], None],
    ) -> bool:
        """Start a batch. False when one is already running."""
        if not requests or (self._job is not None and self._job.isRunning()):
            return False
        self._job = GuidedBatchThread(requests, cancel, 0, 600.0)
        self._job.progressed.connect(on_progress)
        self._job.item_finished.connect(on_item)
        self._job.start()
        return True

    def open_batch(self, paths: list[Path] | None = None) -> None:
        """Open the batch screen and add ``paths`` when they were dropped or chosen."""
        if self._batch is None:
            self._batch = BatchScreen(self, self)
        if paths:
            self._batch.add_paths(paths)
        self._batch.resize(1200, 900)
        self._batch.show()
        self._batch.raise_()

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
            self._stop_simple(tr("enlarge.error.missing"))
            return
        settings = self.job_settings()
        if settings.lossless and settings.image_format != "jxl":
            self._stop_simple(tr("enlarge.error.lossless"))
            return
        if not self.models_ready():
            self.simple.unavailable(tr("enlarge.error.models"))
            self.view.refresh()
            return
        try:
            built = requests_for([source], settings, APP_DISPLAY_NAME)
        except (OSError, ValueError) as exc:
            self._stop_simple(str(exc))
            return
        request, scale = built[0]
        self._fits[request.id] = (source, scale)
        started = self.launch(
            [request],
            lambda: self.simple.phase is not Phase.BUSY,
            self._on_progress,
            self._on_item,
        )
        if not started:
            self._stop_simple(tr("enlarge.batch.busy"))

    def _stop_simple(self, message: str) -> None:
        self.simple.cancel()
        self.view.set_note(message)
        self.view.refresh()

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
        settings = self.job_settings()
        if ok and dst and settings.long_side > 0:
            source, scale = self._fits.get(job_id, (Path(self.simple.prompt), 4))
            try:
                fit_long_side(
                    source,
                    Path(dst),
                    scale,
                    settings.long_side,
                    settings.image_format,
                    settings.quality,
                    settings.lossless and settings.image_format == "jxl",
                )
            except (OSError, ValueError) as exc:
                ok = False
                message = str(exc)
        if ok and dst:
            self._results[job_id] = Path(dst)
            self.simple.finish(job_id)
        else:
            self.simple.block(message.strip() or tr("enlarge.error.failed"))
        self.view.refresh()

    def _retarget_results(self) -> None:
        """Use the shell result slots. Save and Open folder stay. The other two are upscaler actions."""
        self._relabel("moreLike", "enlarge.action.compare", self._open_compare)
        self._relabel("usePrompt", "enlarge.action.again", self._again)

    def _relabel(self, name: str, key: str, handler: Callable[[], None]) -> None:
        button = self.view.root.findChild(QPushButton, name)
        if button is None:
            return
        button.setText(tr(key))
        button.clicked.disconnect()
        button.clicked.connect(_click(handler))

    def _extend_overflow(self) -> None:
        """Add the user actions the shell overflow does not already list.

        Open, Batch, and Credits have no slot on this surface, so they are not
        added here and there is no second toolbar.
        """
        button = self.view.root.findChild(QPushButton, "overflow")
        menu = None if button is None else button.menu()
        if menu is None:
            return
        menu.addSeparator()
        for name, key, handler in (
            ("morePreview", "enlarge.more.preview", self._dry_run),
            ("moreGpu", "enlarge.more.gpu", self._check_gpu),
            ("moreAbout", "enlarge.more.about", self._open_about),
            ("moreHash", "enlarge.more.hash", self._check_hash),
            ("moreWeights", "enlarge.more.weights", self._remove_weights),
        ):
            action = menu.addAction(tr(key))
            action.setObjectName(name)
            action.triggered.connect(_click(handler))

    def _again(self) -> None:
        self.view._generate()  # noqa: SLF001 - the shell's Enlarge button uses this path

    def _open_compare(self) -> None:
        source = Path(self.simple.prompt)
        self._compare = CompareScreen(self, source, self._current_result())
        self._compare.show()

    def _open_credits(self) -> None:
        """Credits stays built. This surface has no slot that can open it."""
        self._credits = show_credits(self, str(self.simple.get("enlarge.custom_models") or ""))

    def _open_about(self) -> None:
        show_report(self, tr("enlarge.about.title"), f"{APP_DISPLAY_NAME}\n{version_label()}")

    def _check_gpu(self) -> None:
        buffer = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
            code = entry._selftest(False)
        text = "\n".join(part for part in (buffer.getvalue().strip(), errors.getvalue().strip()) if part)
        show_report(self, tr("enlarge.more.gpu"), text or str(code))

    def _check_hash(self) -> None:
        show_hash(self, str(self.simple.get("enlarge.model")))

    def _remove_weights(self) -> None:
        remove_weights(self, str(self.simple.get("enlarge.model")))

    def _dry_run(self) -> None:
        show_dry_run(self, Path(self.simple.prompt), self.job_settings())

    def _dress_expert(self) -> None:
        """Give spins room, and show Off when the long side is zero."""
        for spin in self.view.root.findChildren(QSpinBox):
            spin.setMinimumWidth(120)
            if spin.objectName() == "enlarge.long_side":
                spin.setSpecialValueText(tr("enlarge.long_side.off"))

    def _fit_bold_controls(self) -> None:
        """Size selected chips and the Expert toggle the way the shell measures bold text."""
        for button in self.view.root.findChildren(QPushButton):
            chip = button.property("chip") == "true" or button.objectName() == "expertToggle"
            result = button.objectName() in {"moreLike", "savePicture", "openFolder", "usePrompt"}
            if chip or result:
                _fit_bold(button, pad=36 if chip else 40)

    def _paint_wipe(self) -> None:
        """Draw a before/after split on the main result after a run."""
        label = self.view.root.findChild(QLabel, "result")
        if label is None or self.simple.phase is not Phase.READY:
            if self._wipe is not None:
                self._wipe.hide()
            return
        slider = self._wipe_slider(label)
        width = max(label.width(), 2)
        height = max(label.height(), 2)
        split = slider.value() / 1000
        before = _pixmap(Path(self.simple.prompt))
        after = _pixmap(self._current_result())
        label.setPixmap(_wipe_frame(before, after, width, height, split))
        slider.setGeometry(label.geometry())
        slider.show()
        slider.raise_()

    def _wipe_slider(self, label: QLabel) -> QSlider:
        if self._wipe is not None:
            return self._wipe
        parent = label.parentWidget()
        slider = QSlider(Qt.Orientation.Horizontal, parent if parent is not None else label)
        slider.setObjectName("compareSlider")
        slider.setRange(0, 1000)
        slider.setValue(500)
        slider.valueChanged.connect(_click(self._paint_wipe))
        label.installEventFilter(self)
        self._result_label = label
        self._wipe = slider
        return slider

    def _install_full_drop(self) -> None:
        self.setAcceptDrops(True)
        drop = self.view.root.findChild(QWidget, "fileDrop")
        if drop is not None:
            drop.setAcceptDrops(True)
            drop.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt's name
        """Keep the full path when a picture is dropped, and redraw the wipe after a resize."""
        if watched is self._result_label and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._paint_wipe)
            return False
        if event.type() == QEvent.Type.DragEnter and _has_urls(event):
            _accept(event)
            return True
        if event.type() == QEvent.Type.Drop:
            paths = _dropped_paths(event)
            if paths:
                self._take_drop(paths)
                _accept(event)
                return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt's name
        """Accept a drag that contains files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt's name
        """Add dropped files. One file fills the main screen. More than one opens the batch."""
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.toLocalFile()]
        if paths:
            self._take_drop(paths)
            event.acceptProposedAction()

    def _take_drop(self, paths: list[Path]) -> None:
        files = [path for path in paths if path.is_file()]
        folders = [path for path in paths if path.is_dir()]
        if len(paths) == 1 and len(files) == 1 and not folders:
            self.view.set_prompt_text(str(files[0]))
            return
        self.open_batch(paths)


def _click(handler: Callable[[], None]) -> Callable[[], None]:
    """Ignore the bool Qt passes from a clicked or triggered signal."""

    def run(*_args: object) -> None:
        handler()

    return run


def _pixmap(path: Path | None) -> QPixmap:
    """Load a still image. A missing file returns an empty pixmap."""
    if path is None or not path.is_file():
        return QPixmap()
    loaded = QPixmap(str(path))
    if not loaded.isNull():
        return loaded
    return _load(path)


def _wipe_frame(before: QPixmap, after: QPixmap, width: int, height: int, split: float) -> QPixmap:
    """One frame with the original on the left of the divider and the result on the right."""
    image = QImage(max(width, 1), max(height, 1), QImage.Format.Format_RGB32)
    image.fill(QColor("#14141c"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    left = before if not before.isNull() else _wash("#243044", width, height)
    right = after if not after.isNull() else _wash("#c4b49a", width, height)
    rect = image.rect()
    painter.drawPixmap(rect, _cover(left, width, height))
    cut = int(max(0.0, min(split, 1.0)) * width)
    painter.setClipRect(cut, 0, max(0, width - cut), height)
    painter.drawPixmap(rect, _cover(right, width, height))
    painter.setClipping(False)
    painter.setPen(QPen(QColor("#f2f2f5"), 2))
    painter.drawLine(cut, 0, cut, height)
    painter.end()
    return QPixmap.fromImage(image)


def _wash(colour: str, width: int, height: int) -> QPixmap:
    image = QImage(max(width, 1), max(height, 1), QImage.Format.Format_RGB32)
    image.fill(QColor(colour))
    return QPixmap.fromImage(image)


def _cover(pixmap: QPixmap, width: int, height: int) -> QPixmap:
    return pixmap.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )


def _fit_bold(button: QPushButton, *, pad: int) -> None:
    """Reserve the bold label width. Selected chips and Expert mode use weight 600."""
    text = button.text()
    if not text:
        return
    font = QFont(button.font())
    font.setWeight(QFont.Weight.DemiBold)
    advance = QFontMetrics(font).horizontalAdvance(text)
    button.setMinimumWidth(max(advance + pad, 36))


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


def _factor_text(scale: object, double: object, long_side: object = 0) -> str:
    side = as_int(long_side)
    if side > 0:
        return f"{side} px"
    try:
        factor = output_scale(int(str(scale)), bool(double))
    except (TypeError, ValueError):
        return ""
    return f"{factor}×"


def _accept(event: QEvent) -> None:
    accept = getattr(event, "acceptProposedAction", None)
    if callable(accept):
        accept()


def _has_urls(event: QEvent) -> bool:
    mime = getattr(event, "mimeData", None)
    if not callable(mime):
        return False
    return bool(mime().hasUrls())


def _dropped_paths(event: QEvent) -> list[Path]:
    mime = getattr(event, "mimeData", None)
    if not callable(mime):
        return []
    return [Path(url.toLocalFile()) for url in mime().urls() if url.toLocalFile()]


def _licence_line(model_id: str) -> str:
    model = load_builtin(release=False).get(model_id)
    if model is None:
        return ""
    licence = model.licence.strip()
    author = model.author.strip()
    if author and licence:
        return f"{author} · {licence}"
    return licence or author


def _models_ready() -> bool:
    try:
        return ready("enlarge")
    except (OSError, RuntimeError, ValueError):
        return False


def _gpu_warning() -> str:
    try:
        report = gpu_capability()
    except (OSError, RuntimeError, ValueError):
        return ""
    if report.get("status") == "warn":
        return tr("enlarge.gpu.warn")
    return ""
