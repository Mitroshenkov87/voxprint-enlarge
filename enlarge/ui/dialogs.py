"""Settings and the credits screen. Credits are built from the registry, not a hand-written list."""
from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from enlarge.appinfo import APP_DISPLAY_NAME, STATUS, version_label
from enlarge.suite import gpu, resolved_models_dir, set_gpu, set_models_dir
from enlarge.ui.theme import build_style
from voxprint_engine.registry.schema import ModelManifest


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _link(url: str, label: str) -> str:
    if url.startswith("https://") or url.startswith("http://"):
        return f'<a href="{_esc(url)}">{_esc(label)}</a>'
    return _esc(label)


class SettingsDialog(QDialog):
    """GPU and folders. GPU and the shared models folder are written to suite.json."""

    def __init__(self, parent: QWidget, settings: dict[str, object]) -> None:
        super().__init__(parent)
        self.setObjectName("root")
        self.setWindowTitle("Settings")
        self.setStyleSheet(build_style(False))
        self.setMinimumWidth(560)
        self.result_settings = dict(settings)
        self._gpu = gpu()
        self._models = resolved_models_dir()
        self._models_touched = False

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)
        title = QLabel("Settings")
        title.setObjectName("title")
        root.addWidget(title)
        hint = QLabel("GPU and the models folder are shared with the other Voxprint apps (suite.json).")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        card = QFrame()
        card.setObjectName("card")
        form = QVBoxLayout(card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(10)

        form.addWidget(self._label("GPU"))
        self.gpu_box = QComboBox()
        for label, value in (("Auto", "auto"), ("CPU", "cpu"), ("CUDA:0", "cuda:0"), ("CUDA:1", "cuda:1")):
            self.gpu_box.addItem(label, value)
        current = self.gpu_box.findData(self._gpu)
        self.gpu_box.setCurrentIndex(current if current >= 0 else 0)
        form.addWidget(self.gpu_box)

        form.addWidget(self._label("Tile size"))
        self.tile_box = QComboBox()
        self.tile_box.addItem("Auto (from free VRAM)", "auto")
        for size in ("128", "256", "512", "768"):
            self.tile_box.addItem(f"{size} px", size)
        tile = str(settings.get("tile") or "auto")
        index = self.tile_box.findData(tile)
        self.tile_box.setCurrentIndex(index if index >= 0 else 0)
        form.addWidget(self.tile_box)
        tile_hint = QLabel("Auto follows the suite rule: free VRAM minus max(2 GB, 8% of the card).")
        tile_hint.setObjectName("hint")
        tile_hint.setWordWrap(True)
        form.addWidget(tile_hint)

        form.addWidget(self._label("Output folder"))
        self.output_edit = QLineEdit(str(settings.get("output_dir") or ""))
        self.output_edit.setPlaceholderText("Empty: save beside each image")
        form.addLayout(self._browse_row(self.output_edit, self._pick_output, self.output_edit.clear))

        form.addWidget(self._label("Custom models folder"))
        self.custom_edit = QLineEdit(str(settings.get("custom_models_dir") or ""))
        self.custom_edit.setPlaceholderText("Extra manifest JSON files")
        form.addLayout(self._browse_row(self.custom_edit, self._pick_custom, self.custom_edit.clear))

        form.addWidget(self._label("Shared models folder"))
        self.models_edit = QLineEdit(str(self._models))
        form.addLayout(self._browse_row(self.models_edit, self._pick_models, self._reset_models))
        models_hint = QLabel("Weights will live here. Reset goes back to the default Voxprint models folder.")
        models_hint.setObjectName("hint")
        models_hint.setWordWrap(True)
        form.addWidget(models_hint)

        self.error = QLabel("")
        self.error.setObjectName("warn")
        self.error.setWordWrap(True)
        form.addWidget(self.error)
        root.addWidget(card)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        root.addLayout(buttons)

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectiontitle")
        return label

    def _browse_row(self, edit: QLineEdit, on_browse, on_clear) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(on_browse)
        clear = QPushButton("Clear")
        clear.clicked.connect(on_clear)
        row.addWidget(browse)
        row.addWidget(clear)
        return row

    def _pick_dir(self, edit: QLineEdit) -> None:
        start = edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Choose a folder", start)
        if chosen:
            edit.setText(chosen)

    def _pick_output(self) -> None:
        self._pick_dir(self.output_edit)

    def _pick_custom(self) -> None:
        self._pick_dir(self.custom_edit)

    def _pick_models(self) -> None:
        self._models_touched = True
        self._pick_dir(self.models_edit)

    def _reset_models(self) -> None:
        self._models_touched = True
        self.models_edit.clear()

    def _absolute(self, text: str, label: str) -> Path | None:
        if not text.strip():
            return None
        folder = Path(text.strip())
        if not folder.is_absolute():
            self.error.setText(f"{label} must be an absolute path, or empty.")
            return None
        return folder

    def _save(self) -> None:
        self.error.setText("")
        output = self._absolute(self.output_edit.text(), "Output folder")
        if self.error.text():
            return
        custom = self._absolute(self.custom_edit.text(), "Custom models folder")
        if self.error.text():
            return
        models_text = self.models_edit.text().strip()
        models: Path | None
        if not models_text:
            models = None
        else:
            models = self._absolute(models_text, "Shared models folder")
            if self.error.text():
                return
        choice = str(self.gpu_box.currentData())
        try:
            set_gpu(choice)
            if self._models_touched or models_text != str(self._models):
                set_models_dir(models)
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))
            return
        self.result_settings["tile"] = str(self.tile_box.currentData())
        self.result_settings["output_dir"] = "" if output is None else str(output)
        self.result_settings["custom_models_dir"] = "" if custom is None else str(custom)
        self.accept()


class CreditsDialog(QDialog):
    """About this build, plus every CC-BY model in the registry."""

    def __init__(self, parent: QWidget, credits: list[ModelManifest], warnings: list[str]) -> None:
        super().__init__(parent)
        self.setObjectName("root")
        self.setWindowTitle("Credits")
        self.setStyleSheet(build_style(False))
        self.resize(640, 560)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        title = QLabel(APP_DISPLAY_NAME)
        title.setObjectName("title")
        root.addWidget(title)
        sub = QLabel(f"{STATUS}  ·  {version_label()}  ·  images only")
        sub.setObjectName("subtitle")
        root.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QLabel()
        body.setObjectName("creditsBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        body.setOpenExternalLinks(True)
        palette = body.palette()
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#f2f2f5"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#f2f2f5"))
        palette.setColor(QPalette.ColorRole.Link, QColor("#93c5fd"))
        body.setPalette(palette)
        body.setStyleSheet("color: #f2f2f5; background: transparent; font-size: 14px;")
        body.setText(self._html(credits, warnings))
        holder = QWidget()
        holder.setObjectName("content")
        lay = QVBoxLayout(holder)
        lay.addWidget(body)
        lay.addStretch(1)
        scroll.setWidget(holder)
        root.addWidget(scroll, 1)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        root.addLayout(row)

    def _html(self, credits: list[ModelManifest], warnings: list[str]) -> str:
        parts = [
            "<p>Voxprint AI Enlarge is free software under the Apache License 2.0. "
            "Copyright 2026 Aleksandr Mitroshenkov. This build does not download or run "
            "neural weights. The Run button resizes with Lanczos so the rest of the flow "
            "can be tried. Model names below are the ones the registry will load later.</p>",
            "<p><b>CC-BY models</b> — credit is required when these weights are used.</p>",
        ]
        if not credits:
            parts.append("<p>No CC-BY models are registered.</p>")
        for model in credits:
            parts.append(
                "<p><b>" + _esc(model.name) + "</b><br>"
                + _esc(model.attribution) + "<br>"
                + _esc(model.author) + " · " + _esc(model.licence) + "<br>"
                + _link(model.source_url, model.source_url) + "</p>"
            )
        if warnings:
            parts.append("<p><b>Custom folder</b></p>")
            for warning in warnings:
                parts.append("<p>" + _esc(warning) + "</p>")
        parts.append(
            "<p>Other registered models use Apache-2.0 or BSD-3-Clause and are listed in the "
            "model step when they can be selected. Non-commercial weights are rejected and do not appear here.</p>"
        )
        return "\n".join(parts)
