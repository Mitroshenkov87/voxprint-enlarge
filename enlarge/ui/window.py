"""Main window: Images, Model, Output, Run. Heavy work stays in a worker process."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QIcon, QPixmap, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from enlarge import platform_win, settings, suite
from enlarge.appinfo import APP_DISPLAY_NAME, resource_dir, status_label, version_label
from enlarge.images import collect, plan_outputs
from enlarge.queue_state import dump_queue, load_queue, still_open
from enlarge.ui import pages
from enlarge.ui.dialogs import CreditsDialog, SettingsDialog
from enlarge.ui.jobs import BatchThread
from enlarge.ui.models import DownloadThread
from enlarge.ui.theme import build_style
from voxprint_engine.formats import DEFAULT_FORMAT, DEFAULT_QUALITY
from voxprint_engine.hardware import assess_gpu
from voxprint_engine.registry import DEFAULT_MODEL_ID, combined_catalog
from voxprint_engine.registry.catalog import Catalog
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.store import StoreError, delete_weight, weight_report
from voxprint_engine.vram import choose_tile, device_index, measure, overlap_for
from voxprint_engine.worker import UpscaleRequest

MIN_W, MIN_H = 880, 620
STEPS = ("Images", "Model", "Output", "Run")
_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.avif *.jxl *.bmp *.tif *.tiff)"


class MainWindow(QWidget):
    """The five-step window: Images, Model, Output, Run, and the menus."""
    def __init__(self) -> None:
        """Build the pages and restore a saved queue."""
        super().__init__()
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setObjectName("root")
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.backdrop = "plain"
        self.settings = settings.load()
        self.catalog = self._load_catalog()
        self.sources: list[Path] = []
        self.rows: list[dict[str, object]] = []
        self.job: BatchThread | None = None
        self.download: DownloadThread | None = None
        self._cancel = False
        self._loading = False
        self._snapshot = measure(suite.gpu())
        self._build()
        self.setStyleSheet(build_style(False))
        self._apply_settings()
        self._restore_queue()
        self._fit()

    def _load_catalog(self) -> Catalog:
        custom = str(self.settings.get("custom_models_dir") or "").strip()
        folder = Path(custom) if custom else None
        return combined_catalog(folder, release=True)

    def _build(self) -> None:
        body = QVBoxLayout(self)
        body.setContentsMargins(22, 16, 22, 14)
        body.setSpacing(10)
        head = QHBoxLayout()
        mark = QLabel()
        icon_file = resource_dir() / "assets" / "enlarge.png"
        pixmap = QPixmap(str(icon_file))
        if not pixmap.isNull():
            mark.setPixmap(pixmap.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        head.addWidget(mark)
        self.lbl_title = QLabel(APP_DISPLAY_NAME)
        self.lbl_title.setObjectName("title")
        head.addWidget(self.lbl_title)
        head.addStretch(1)
        self.btn_settings = QPushButton("Settings")
        self.btn_credits = QPushButton("Credits")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_credits.clicked.connect(self.open_credits)
        head.addWidget(self.btn_settings)
        head.addWidget(self.btn_credits)
        body.addLayout(head)

        self.lbl_tagline = QLabel("Offline image upscaling for the Voxprint suite. Nothing is uploaded.")
        self.lbl_tagline.setObjectName("subtitle")
        self.lbl_tagline.setWordWrap(True)
        body.addWidget(self.lbl_tagline)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.step_buttons: list[QPushButton] = []
        for index, name in enumerate(STEPS):
            button = QPushButton(f"{index + 1}. {name}")
            button.setObjectName("step")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, number=index: self.go(number))
            self.step_buttons.append(button)
            bar.addWidget(button, 1)
        body.addLayout(bar)

        self.images_page = pages.ImagesPage()
        self.model_page = pages.ModelPage()
        self.output_page = pages.OutputPage()
        self.run_page = pages.RunPage()
        self.stack = QStackedWidget()
        self.stack.setObjectName("central")
        for page in (self.images_page, self.model_page, self.output_page, self.run_page):
            self.stack.addWidget(page)
        body.addWidget(self.stack, 1)
        self.lbl_footer = QLabel(
            f"Images only  ·  {status_label()}  ·  TensorRT, CUDA if that engine does not build  ·  {version_label()}"
        )
        self.lbl_footer.setObjectName("footer")
        body.addWidget(self.lbl_footer)

        self.images_page.add_files.connect(self.choose_files)
        self.images_page.add_folder.connect(self.choose_folder)
        self.images_page.remove_selected.connect(self.remove_selected)
        self.images_page.clear_all.connect(self.clear_images)
        self.model_page.models.currentItemChanged.connect(lambda *_args: self._on_model())
        self.model_page.download_clicked.connect(self.download_model)
        self.model_page.check_clicked.connect(self.check_model)
        self.model_page.remove_clicked.connect(self.remove_model)
        self.output_page.scale_x2.clicked.connect(lambda: self._set_scale(2))
        self.output_page.scale_x4.clicked.connect(lambda: self._set_scale(4))
        self.output_page.double.toggled.connect(lambda _on: self._on_output_changed())
        self.output_page.format_box.currentIndexChanged.connect(lambda _i: self._on_output_changed())
        self.output_page.lossless.toggled.connect(lambda _on: self._on_output_changed())
        self.output_page.quality.valueChanged.connect(self._on_quality)
        self.output_page.btn_output.clicked.connect(self.choose_output_folder)
        self.output_page.btn_output_clear.clicked.connect(self.clear_output_folder)
        self.output_page.output_edit.editingFinished.connect(self._on_output_changed)
        self.run_page.run_clicked.connect(self.start_run)
        self.run_page.cancel_clicked.connect(self.cancel_run)
        self.run_page.zoom_in.connect(lambda: self._zoom(1.25))
        self.run_page.zoom_out.connect(lambda: self._zoom(0.8))
        self.run_page.zoom_fit.connect(self._zoom_fit)
        self.run_page.table.currentCellChanged.connect(self._on_queue_row)
        self.run_page.table.verticalHeader().setVisible(False)
        self.setWindowTitle(APP_DISPLAY_NAME)
        self._set_window_icon()

    def _set_window_icon(self) -> None:
        for name in ("enlarge.ico", "enlarge.png"):
            path = resource_dir() / "assets" / name
            if path.is_file():
                icon = QIcon(str(path))
                self.setWindowIcon(icon)
                app = QApplication.instance()
                if isinstance(app, QApplication):
                    app.setWindowIcon(icon)
                return

    def _fit(self) -> None:
        width, height = 1120, 760
        try:
            screen = self.screen() or QApplication.primaryScreen()
            if screen is not None:
                area = screen.availableGeometry()
                width, height = min(width, int(area.width() * 0.94)), min(height, int(area.height() * 0.90))
        except RuntimeError:
            pass
        self.setMinimumSize(MIN_W, MIN_H)
        self.resize(max(width, MIN_W), max(height, MIN_H))

    def _apply_settings(self) -> None:
        self._loading = True
        try:
            self.output_page.set_scale(int(self.settings["scale"]))
            self.output_page.double.setChecked(bool(self.settings["double_upscale"]))
            self.output_page.fill_formats()
            self.output_page.apply_output(
                str(self.settings["format"]),
                int(self.settings["quality"]),
                bool(self.settings.get("lossless")),
            )
            self.output_page.output_edit.setText(str(self.settings.get("output_dir") or ""))
            self.images_page.subfolders.setChecked(bool(self.settings["include_subfolders"]))
            self.model_page.fill(self.catalog.selectable(), str(self.settings.get("model_id") or DEFAULT_MODEL_ID))
        finally:
            self._loading = False
        self._on_quality(self.output_page.quality.value())
        self._update_tile_label()
        self._refresh_gpu_warning()
        self.go(0)

    def _persist(self) -> None:
        if self._loading:
            return
        self.settings = settings.save({
            "model_id": self.model_page.current_id() or self.settings.get("model_id") or DEFAULT_MODEL_ID,
            "scale": self.output_page.scale(),
            "format": self.output_page.image_format(),
            "quality": self.output_page.quality.value(),
            "lossless": self.output_page.lossless.isChecked() and self.output_page.image_format() == "jxl",
            "double_upscale": self.output_page.double.isChecked(),
            "tile": self.settings.get("tile") or "auto",
            "output_dir": self.output_page.output_edit.text().strip(),
            "custom_models_dir": self.settings.get("custom_models_dir") or "",
            "include_subfolders": self.images_page.subfolders.isChecked(),
        })

    def _output_folder(self) -> Path | None:
        text = self.output_page.output_edit.text().strip()
        if not text:
            return None
        folder = Path(text)
        if not folder.is_absolute():
            return None
        return folder

    def _user_tile(self) -> int:
        tile = str(self.settings.get("tile") or "auto")
        if tile == "auto":
            return 0
        try:
            return int(tile)
        except ValueError:
            return 0

    def _refresh_gpu_warning(self) -> None:
        self.output_page.gpu_warning.setText(assess_gpu(device_index(suite.gpu())).warning)

    def _update_tile_label(self) -> None:
        model = self.catalog.get(self.model_page.current_id())
        model_tile = model.tile if model is not None else 0
        tile = choose_tile(self._snapshot.budget_gb, model_tile, self._user_tile())
        mode = "auto" if self._user_tile() == 0 else f"fixed {self._user_tile()}"
        self.output_page.tile_label.setText(
            f"Tile {tile} px, overlap {overlap_for(tile)} px ({mode}). "
            f"VRAM budget {self._snapshot.budget_gb:.1f} GB ({self._snapshot.source})."
        )

    def go(self, index: int) -> None:
        """Switch to the step at this index."""
        index = max(0, min(index, len(STEPS) - 1))
        self.stack.setCurrentIndex(index)
        for number, button in enumerate(self.step_buttons):
            button.setChecked(number == index)
        if index == 3:
            self._rebuild_rows()

    def choose_files(self) -> None:
        """Add images from a file dialog."""
        names, _selected = QFileDialog.getOpenFileNames(self, "Add images", str(Path.home()), _IMAGE_FILTER)
        if names:
            self.add_paths([Path(name) for name in names])

    def choose_folder(self) -> None:
        """Add a folder, including subfolders when that option is on."""
        folder = QFileDialog.getExistingDirectory(self, "Add a folder", str(Path.home()))
        if folder:
            self.add_paths([Path(folder)])

    def add_paths(self, paths: list[Path]) -> None:
        """Add these files or folders to the image list."""
        found, warnings = collect(paths, recursive=self.images_page.subfolders.isChecked())
        have = {self._key(path) for path in self.sources}
        for path in found:
            key = self._key(path)
            if key not in have:
                self.sources.append(path)
                have.add(key)
        self.images_page.note.setText(" ".join(warnings[:4]))
        self._sync_list()

    def remove_selected(self) -> None:
        """Remove the selected rows from the image list."""
        indexes = sorted({item.row() for item in self.images_page.files.selectedIndexes()}, reverse=True)
        for index in indexes:
            if 0 <= index < len(self.sources):
                del self.sources[index]
        self._sync_list()

    def clear_images(self) -> None:
        """Clear the image list."""
        self.sources.clear()
        self.images_page.note.setText("")
        self._sync_list()

    def _sync_list(self) -> None:
        self.images_page.files.clear()
        for path in self.sources:
            self.images_page.files.addItem(f"{path.name}   {path.parent}")
        count = len(self.sources)
        self.images_page.count.setText("No images yet" if count == 0 else f"{count} image" + ("s" if count != 1 else ""))
        if not self._running():
            self._rebuild_rows()
        self.run_page.btn_run.setEnabled(count > 0 and not self._running())

    def _rebuild_rows(self) -> None:
        if self._running():
            return
        try:
            planned = plan_outputs(
                self.sources,
                self._output_folder(),
                self.output_page.scale(),
                self.output_page.image_format(),
                self.output_page.double.isChecked(),
            )
        except ValueError as exc:
            self.run_page.status.setText(str(exc))
            return
        previous = {self._key(Path(str(row["src"]))): row for row in self.rows}
        rebuilt: list[dict[str, object]] = []
        for src, dst in zip(self.sources, planned, strict=True):
            old = previous.get(self._key(src))
            status = str(old.get("status") or "") if old is not None else ""
            if old is not None and status in {"Done", "Failed", "Interrupted", "Cancelled"}:
                dst = Path(str(old["dst"]))
            same_file = old is not None and Path(str(old["dst"])) == dst
            keep = same_file and status in {"Done", "Failed", "Interrupted", "Cancelled"}
            if keep and status == "Done" and not dst.is_file():
                keep = False
            message = str(old.get("message") or "") if keep and old is not None else ""
            rebuilt.append({
                "id": str(old["id"]) if old is not None else uuid.uuid4().hex,
                "src": src,
                "dst": dst,
                "status": status if keep else "Waiting",
                "message": message,
            })
        self.rows = rebuilt
        self._fill_table()

    def _fill_table(self) -> None:
        table = self.run_page.table
        table.setRowCount(len(self.rows))
        for index, row in enumerate(self.rows):
            table.setItem(index, 0, QTableWidgetItem(Path(str(row["src"])).name))
            table.setItem(index, 1, QTableWidgetItem(str(row["status"])))
            table.setItem(index, 2, QTableWidgetItem(str(Path(str(row["dst"])).name)))
        self.run_page.btn_run.setEnabled(bool(self.rows) and not self._running())

    def _on_model(self) -> None:
        model = self.catalog.get(self.model_page.current_id())
        if model is None:
            self.model_page.detail.setText("")
        else:
            self.model_page.detail.setText(f"{model.author}  ·  {model.licence}  ·  {model.filename}")
        self._refresh_model_file()
        self._update_tile_label()
        self._persist()

    def _set_scale(self, scale: int) -> None:
        self.output_page.set_scale(scale)
        self._on_output_changed()

    def _on_quality(self, value: int) -> None:
        image_format = self.output_page.image_format()
        box = self.output_page.lossless
        box.setEnabled(image_format == "jxl")
        if image_format != "jxl" and box.isChecked():
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
        lossless = image_format == "jxl" and box.isChecked()
        self.output_page.quality.setEnabled(not lossless)
        if lossless:
            self.output_page.quality_label.setText("Lossless JPEG XL ignores the quality slider")
        else:
            self.output_page.quality_label.setText(f"Quality {value}")
        if not self._loading:
            self._persist()

    def _on_output_changed(self) -> None:
        text = self.output_page.output_edit.text().strip()
        if text and not Path(text).is_absolute():
            self.output_page.tile_label.setText("Output folder must be an absolute path, or left empty.")
            return
        self._on_quality(self.output_page.quality.value())
        self._update_tile_label()
        self._persist()
        self._rebuild_rows()

    def choose_output_folder(self) -> None:
        """Pick a folder for the enlarged files."""
        start = self.output_page.output_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Output folder", start)
        if folder:
            self.output_page.output_edit.setText(folder)
            self._on_output_changed()

    def clear_output_folder(self) -> None:
        """Save beside each source again."""
        self.output_page.output_edit.clear()
        self._on_output_changed()

    def start_run(self) -> None:
        """Start the batch for the current list, model, and output settings."""
        if self._running() or not self.rows:
            self.run_page.status.setText("Add an image first.")
            return
        self._persist()
        model_id = self.model_page.current_id()
        model = self.catalog.get(model_id)
        if model is None or not model.selectable:
            self.run_page.status.setText("Pick a model from the list.")
            return
        self._snapshot = measure(suite.gpu())
        self._update_tile_label()
        requests: list[UpscaleRequest] = []
        for row in self.rows:
            if not still_open(row):
                continue
            row["status"] = "Waiting"
            row["message"] = ""
            requests.append(UpscaleRequest(
                id=str(row["id"]),
                src=str(row["src"]),
                dst=str(row["dst"]),
                scale=self.output_page.scale(),
                format=self.output_page.image_format(),
                quality=int(self.output_page.quality.value()),
                lossless=self.output_page.lossless.isChecked() and self.output_page.image_format() == "jxl",
                model_id=model.id,
                tile=self._user_tile(),
                model_tile=model.tile,
                double=self.output_page.double.isChecked(),
                device=suite.gpu(),
                engine="auto",
                custom_models_dir=str(self.settings.get("custom_models_dir") or ""),
            ))
        if not requests:
            self.run_page.status.setText("Everything in this queue is already done.")
            return
        self._fill_table()
        self._save_queue()
        self._cancel = False
        self.job = BatchThread(requests, lambda: self._cancel)
        self.job.progressed.connect(self._on_progress)
        self.job.item_finished.connect(self._on_item)
        self.job.finished.connect(self._on_thread_done)
        self._set_busy(True)
        self.run_page.status.setText("Running…")
        self.go(3)
        self.job.start()

    def cancel_run(self) -> None:
        """Cancel the image that is running."""
        self._cancel = True
        if self.job is not None:
            self.job.cancel()
        self.run_page.btn_cancel.setEnabled(False)
        self.run_page.status.setText("Cancelling the current image…")

    def _on_progress(self, job_id: str, fraction: float, message: str) -> None:
        for row in self.rows:
            if str(row["id"]) != job_id or str(row["status"]) in {"Done", "Failed", "Cancelled"}:
                continue
            row["status"] = f"{int(round(fraction * 100))}%"
            row["message"] = message
            self._fill_table()
            break
        self.run_page.status.setText(message)

    def _on_item(self, job_id: str, ok: bool, message: str, dst: str) -> None:
        for index, row in enumerate(self.rows):
            if str(row["id"]) != job_id:
                continue
            if message == "cancelled":
                row["status"] = "Cancelled"
            else:
                row["status"] = "Done" if ok else "Failed"
            row["message"] = message
            if dst:
                row["dst"] = Path(dst)
            self._fill_table()
            if ok:
                self.run_page.table.selectRow(index)
                self._show_row(index)
            break
        self._save_queue()
        finished = sum(1 for row in self.rows if row["status"] in ("Done", "Failed", "Cancelled"))
        self.run_page.status.setText(f"{finished} of {len(self.rows)}  ·  {message}")

    def _on_thread_done(self) -> None:
        self.job = None
        self._set_busy(False)

    def _on_queue_row(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        self._show_row(row)

    def _show_row(self, index: int) -> None:
        if index < 0 or index >= len(self.rows):
            return
        row = self.rows[index]
        src = Path(str(row["src"]))
        dst = Path(str(row["dst"]))
        after = dst if row["status"] == "Done" and dst.is_file() else None
        self.run_page.compare.set_pair(src, after)
        self.run_page.zoom_text.setText(self.run_page.compare.zoom_text())

    def _zoom(self, factor: float) -> None:
        self.run_page.compare.zoom_by(factor)
        self.run_page.zoom_text.setText(self.run_page.compare.zoom_text())

    def _zoom_fit(self) -> None:
        self.run_page.compare.zoom_fit()
        self.run_page.zoom_text.setText(self.run_page.compare.zoom_text())

    def open_settings(self) -> None:
        """Open the settings dialog."""
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        self.settings = settings.save(dialog.result_settings)
        self.catalog = self._load_catalog()
        self._snapshot = measure(suite.gpu())
        self._loading = True
        try:
            self.output_page.output_edit.setText(str(self.settings.get("output_dir") or ""))
            self.output_page.fill_formats()
            self.output_page.apply_output(
                str(self.settings.get("format") or DEFAULT_FORMAT),
                int(self.settings.get("quality") or DEFAULT_QUALITY),
                bool(self.settings.get("lossless")),
            )
            self.model_page.fill(self.catalog.selectable(), str(self.settings.get("model_id") or DEFAULT_MODEL_ID))
        finally:
            self._loading = False
        self._on_quality(self.output_page.quality.value())
        self._update_tile_label()
        self._refresh_gpu_warning()
        self._rebuild_rows()

    def open_credits(self) -> None:
        """Open the credits dialog."""
        CreditsDialog(self, list(self.catalog.models), self.catalog.warnings).exec()

    def _current_model(self) -> ModelManifest | None:
        model = self.catalog.get(self.model_page.current_id())
        if model is None or not model.selectable:
            return None
        return model

    def _refresh_model_file(self) -> None:
        model = self._current_model()
        if model is None:
            self.model_page.file_status.setText("")
            return
        _state, detail = weight_report(model, suite.resolved_models_dir())
        self.model_page.file_status.setText(f"{model.filename}  ·  {detail}")

    def check_model(self) -> None:
        """Compare the selected weight file with its pinned sha256."""
        self._refresh_model_file()

    def download_model(self) -> None:
        """Download the selected model into the shared models folder."""
        model = self._current_model()
        if model is None or (self.download is not None and self.download.isRunning()):
            return
        self.model_page.btn_download.setEnabled(False)
        thread = DownloadThread(model, suite.resolved_models_dir())
        self.download = thread
        thread.progressed.connect(lambda _fraction, message: self.model_page.file_status.setText(message))
        thread.succeeded.connect(self._download_done)
        thread.failed.connect(self._download_failed)
        thread.start()

    def _download_done(self, path: str) -> None:
        self.model_page.btn_download.setEnabled(True)
        self.model_page.file_status.setText(f"saved {Path(path).name}  ·  sha256 matches")

    def _download_failed(self, message: str) -> None:
        self.model_page.btn_download.setEnabled(True)
        self.model_page.file_status.setText(message)

    def remove_model(self) -> None:
        """Delete the weight file when no other Voxprint app still uses it."""
        model = self._current_model()
        if model is None:
            return
        answer = QMessageBox.question(
            self,
            APP_DISPLAY_NAME,
            f"Remove {model.filename} from the shared models folder?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_weight(model, suite.resolved_models_dir())
        except (OSError, StoreError) as exc:
            QMessageBox.warning(self, APP_DISPLAY_NAME, str(exc))
            return
        self._refresh_model_file()

    def _save_queue(self) -> None:
        if not self.rows:
            return
        try:
            dump_queue(self.rows)
        except OSError:
            return

    def _restore_queue(self) -> None:
        saved = load_queue()
        if not saved:
            return
        self.sources = [Path(str(row["src"])) for row in saved if Path(str(row["src"])).is_file()]
        self.rows = [
            {
                "id": str(row["id"]) or uuid.uuid4().hex,
                "src": Path(str(row["src"])),
                "dst": Path(str(row["dst"])),
                "status": str(row["status"]),
                "message": str(row["message"]),
            }
            for row in saved
            if Path(str(row["src"])).is_file()
        ]
        self._sync_list()
        if any(still_open(row) for row in self.rows):
            self.run_page.status.setText("A previous queue was restored. Run continues the unfinished images.")

    def _running(self) -> bool:
        return self.job is not None and self.job.isRunning()

    def _set_busy(self, busy: bool) -> None:
        self.images_page.btn_files.setEnabled(not busy)
        self.images_page.btn_folder.setEnabled(not busy)
        self.images_page.btn_remove.setEnabled(not busy)
        self.images_page.btn_clear.setEnabled(not busy)
        self.run_page.btn_run.setEnabled(not busy and bool(self.rows))
        self.run_page.btn_cancel.setEnabled(busy)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt name
        """Accept a drag that contains files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt name
        """Add the dropped files or folders."""
        self.add_paths([Path(url.toLocalFile()) for url in event.mimeData().urls()])
        event.acceptProposedAction()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt name
        """Finish the first layout once the window is on screen."""
        super().showEvent(event)
        self.setWindowTitle(APP_DISPLAY_NAME)
        if sys.platform == "win32" and self.backdrop == "plain":
            self.backdrop = platform_win.apply_backdrop(int(self.winId()))
        glass = self.backdrop == "acrylic"
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, glass)
        self.setStyleSheet(build_style(glass))
        if not glass:
            self.setAutoFillBackground(True)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt name
        """Save the queue, then close."""
        self._cancel = True
        if self.job is not None and self.job.isRunning():
            self.job.cancel()
            self.job.wait(8000)
        super().closeEvent(event)

    @staticmethod
    def _key(path: Path) -> str:
        try:
            return str(path.resolve())
        except OSError:
            return str(path)
