"""The four steps: Images, Model, Output, Run."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from enlarge.ui.compare import CompareView
from voxprint_engine.registry.schema import ModelManifest


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    return frame, layout


class ImagesPage(QWidget):
    add_files = Signal()
    add_folder = Signal()
    remove_selected = Signal()
    clear_all = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card, layout = _card()
        title = QLabel("Pictures")
        title.setObjectName("sectiontitle")
        layout.addWidget(title)
        hint = QLabel("Add still images, or a folder of them. Video files are skipped.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.btn_files = QPushButton("Add images")
        self.btn_files.setObjectName("primary")
        self.btn_folder = QPushButton("Add folder")
        self.btn_remove = QPushButton("Remove")
        self.btn_clear = QPushButton("Clear")
        self.btn_files.clicked.connect(self.add_files.emit)
        self.btn_folder.clicked.connect(self.add_folder.emit)
        self.btn_remove.clicked.connect(self.remove_selected.emit)
        self.btn_clear.clicked.connect(self.clear_all.emit)
        for button in (self.btn_files, self.btn_folder, self.btn_remove, self.btn_clear):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self.subfolders = QCheckBox("Include subfolders")
        self.subfolders.setChecked(True)
        layout.addWidget(self.subfolders)
        self.files = QListWidget()
        layout.addWidget(self.files, 1)
        self.count = QLabel("No images yet")
        self.count.setObjectName("status")
        layout.addWidget(self.count)
        self.note = QLabel("")
        self.note.setObjectName("warn")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        root.addWidget(card, 1)


class ModelPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card, layout = _card()
        title = QLabel("Model")
        title.setObjectName("sectiontitle")
        layout.addWidget(title)
        hint = QLabel("Licence-safe models from the registry. The choice is sent with the job. Weights are not loaded in this build.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.models = QListWidget()
        self.models.setWordWrap(True)
        self.models.setSpacing(2)
        layout.addWidget(self.models, 1)
        self.detail = QLabel("")
        self.detail.setObjectName("status")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        root.addWidget(card, 1)

    def fill(self, models: list[ModelManifest], selected: str) -> None:
        self.models.clear()
        for model in models:
            vram = f"~{model.vram_mb} MB VRAM hint" if model.vram_mb else "VRAM hint not set"
            text = f"{model.name}\n{model.summary}\n{model.arch}  ·  native {model.scale}x  ·  {model.licence}  ·  {vram}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, model.id)
            item.setToolTip(model.attribution)
            self.models.addItem(item)
            if model.id == selected:
                self.models.setCurrentItem(item)
        if self.models.currentRow() < 0 and self.models.count():
            self.models.setCurrentRow(0)

    def current_id(self) -> str:
        item = self.models.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")


class OutputPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card, layout = _card()
        title = QLabel("Scale and file")
        title.setObjectName("sectiontitle")
        layout.addWidget(title)

        scale_row = QHBoxLayout()
        self.scale_x2 = QPushButton("x2")
        self.scale_x4 = QPushButton("x4")
        for button in (self.scale_x2, self.scale_x4):
            button.setObjectName("step")
            button.setCheckable(True)
            scale_row.addWidget(button)
        scale_row.addStretch(1)
        layout.addLayout(scale_row)

        self.double = QCheckBox("Double upscale (apply the scale twice)")
        layout.addWidget(self.double)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format"))
        self.format_box = QComboBox()
        self.format_box.addItem("PNG", "png")
        self.format_box.addItem("JPEG", "jpeg")
        self.format_box.addItem("WebP", "webp")
        format_row.addWidget(self.format_box)
        format_row.addStretch(1)
        layout.addLayout(format_row)

        self.quality_label = QLabel("Quality 95")
        layout.addWidget(self.quality_label)
        self.quality = QSlider(Qt.Orientation.Horizontal)
        self.quality.setRange(50, 100)
        self.quality.setValue(95)
        layout.addWidget(self.quality)

        layout.addWidget(QLabel("Output folder"))
        folder_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Empty: save beside each image")
        self.btn_output = QPushButton("Browse")
        self.btn_output_clear = QPushButton("Beside source")
        folder_row.addWidget(self.output_edit, 1)
        folder_row.addWidget(self.btn_output)
        folder_row.addWidget(self.btn_output_clear)
        layout.addLayout(folder_row)

        self.tile_label = QLabel("Tile: Auto")
        self.tile_label.setObjectName("hint")
        self.tile_label.setWordWrap(True)
        layout.addWidget(self.tile_label)
        note = QLabel("This build writes a Lanczos preview at the scale you pick. The neural model is not applied yet.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        root.addWidget(card, 1)

    def scale(self) -> int:
        return 2 if self.scale_x2.isChecked() else 4

    def set_scale(self, scale: int) -> None:
        self.scale_x2.setChecked(scale == 2)
        self.scale_x4.setChecked(scale != 2)

    def image_format(self) -> str:
        return str(self.format_box.currentData() or "png")


class RunPage(QWidget):
    run_clicked = Signal()
    cancel_clicked = Signal()
    zoom_in = Signal()
    zoom_out = Signal()
    zoom_fit = Signal()

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        split = QHBoxLayout()
        queue_card, queue_layout = _card()
        heading = QLabel("Queue")
        heading.setObjectName("sectiontitle")
        queue_layout.addWidget(heading)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Output"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        queue_layout.addWidget(self.table, 1)
        split.addWidget(queue_card, 1)

        view_card, view_layout = _card()
        view_heading = QLabel("Compare")
        view_heading.setObjectName("sectiontitle")
        view_layout.addWidget(view_heading)
        self.compare = CompareView()
        view_layout.addWidget(self.compare, 1)
        zoom = QHBoxLayout()
        self.btn_out = QPushButton("−")
        self.btn_in = QPushButton("+")
        self.btn_fit = QPushButton("Fit")
        self.zoom_text = QLabel("100%")
        self.zoom_text.setObjectName("hint")
        self.btn_out.clicked.connect(self.zoom_out.emit)
        self.btn_in.clicked.connect(self.zoom_in.emit)
        self.btn_fit.clicked.connect(self.zoom_fit.emit)
        for button in (self.btn_out, self.btn_in, self.btn_fit):
            zoom.addWidget(button)
        zoom.addWidget(self.zoom_text)
        zoom.addStretch(1)
        view_layout.addLayout(zoom)
        split.addWidget(view_card, 1)
        root.addLayout(split, 1)

        actions = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.setObjectName("primary")
        self.btn_run.setProperty("big", True)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_run.clicked.connect(self.run_clicked.emit)
        self.btn_cancel.clicked.connect(self.cancel_clicked.emit)
        actions.addWidget(self.btn_run, 1)
        actions.addWidget(self.btn_cancel)
        root.addLayout(actions)
        self.status = QLabel("Add images, pick a model, then run.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
