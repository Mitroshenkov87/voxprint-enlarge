"""Extra screens the shared enlarge surface does not draw.

Batch, compare, credits, the Lucent hand-off, and the process commands live
here. The main screen stays ``surface_for("enlarge")``.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path

import main as entry

from PySide6.QtCore import Qt, QProcess, QThread, Signal
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from enlarge.appinfo import APP_DISPLAY_NAME, status_label, version_label
from enlarge.credits import credit_sections
from enlarge.images import collect
from enlarge.ipc.capabilities import capabilities
from enlarge.ipc.naming import listen_name
from enlarge.ipc.register import register_install, unregister_install
from enlarge.queue_state import dump_queue, load_queue, still_open
from enlarge.ui.compare import CompareView
from enlarge.ui.jobplan import (
    JobSettings,
    command_line,
    describe_job,
    fit_long_side,
    progress_line,
    requests_for,
)
from voxprint_common.i18n import tr
from voxprint_common.paths import state_dir
from voxprint_common.ui.theme import build_shell_style
from voxprint_engine.registry import combined_catalog, load_builtin
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.store import StoreError, delete_weight, weight_report
from voxprint_engine.worker import UpscaleRequest

_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.avif *.jxl *.bmp *.tif *.tiff)"
_STATUS_KEYS = {
    "Waiting": "enlarge.batch.waiting",
    "Done": "enlarge.batch.done",
    "Failed": "enlarge.batch.failed",
    "Cancelled": "enlarge.batch.cancelled",
    "Interrupted": "enlarge.batch.interrupted",
}


class JobHost:
    """The main window, as the extra screens need it."""

    def job_settings(self) -> JobSettings:
        """Current expert and simple values."""
        raise NotImplementedError

    def models_ready(self) -> bool:
        """True when the shared models can run."""
        raise NotImplementedError

    def launch(
        self,
        requests: list[UpscaleRequest],
        cancel: Callable[[], bool],
        on_progress: Callable[[str, float, str], None],
        on_item: Callable[[str, bool, str, str], None],
    ) -> bool:
        """Start a batch. False when one is already running."""
        raise NotImplementedError

    def remember(self, job_id: str, path: Path) -> None:
        """Keep a finished file so Compare and Save can find it."""
        raise NotImplementedError

    def set_subfolders(self, enabled: bool) -> None:
        """Store the shared subfolders switch."""
        raise NotImplementedError


def _screen(dialog: QDialog, name: str, title: str) -> QVBoxLayout:
    dialog.setObjectName(name)
    dialog.setWindowTitle(title)
    dialog.setStyleSheet(build_shell_style(False))
    dialog.resize(1200, 900)
    root = QVBoxLayout(dialog)
    root.setContentsMargins(22, 18, 22, 18)
    root.setSpacing(10)
    heading = QLabel(title)
    heading.setObjectName("title")
    root.addWidget(heading)
    return root


def _button(name: str, key: str, handler: Callable[[], None]) -> QPushButton:
    button = QPushButton(tr(key))
    button.setObjectName(name)
    button.setAutoDefault(False)
    button.clicked.connect(lambda _checked=False: handler())
    return button


class BatchScreen(QDialog):
    """Add images or a folder, then run them with the main screen's settings."""

    def __init__(self, parent: QWidget, host: JobHost) -> None:
        """Build the list, the queue, and the run controls."""
        super().__init__(parent)
        self._host = host
        self._rows: list[dict[str, object]] = []
        self._scales: dict[str, int] = {}
        self._cancel = False
        self._running = False
        self.setAcceptDrops(True)
        root = _screen(self, "batchScreen", tr("enlarge.batch.title"))
        hint = QLabel(tr("enlarge.batch.hint"))
        hint.setWordWrap(True)
        root.addWidget(hint)
        row = QHBoxLayout()
        for name, key, handler in (
            ("batchAdd", "enlarge.batch.add", self._choose_files),
            ("batchFolder", "enlarge.batch.folder", self._choose_folder),
            ("batchRemove", "enlarge.batch.remove", self._remove),
            ("batchClear", "enlarge.batch.clear", self._clear),
        ):
            row.addWidget(_button(name, key, handler))
        row.addStretch(1)
        root.addLayout(row)
        self._subfolders = QCheckBox(tr("enlarge.batch.subfolders"))
        self._subfolders.setObjectName("batchSubfolders")
        self._subfolders.setChecked(host.job_settings().subfolders)
        root.addWidget(self._subfolders)
        self._count = QLabel(tr("enlarge.batch.empty"))
        self._count.setObjectName("batchCount")
        root.addWidget(self._count)
        self._table = QTableWidget(0, 3)
        self._table.setObjectName("batchTable")
        self._table.setHorizontalHeaderLabels(
            [tr("enlarge.batch.file"), tr("enlarge.batch.status"), tr("enlarge.batch.output")]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.currentCellChanged.connect(self._show_row)
        root.addWidget(self._table, 1)
        self._compare = CompareView()
        self._compare.setObjectName("batchCompare")
        root.addWidget(self._compare, 1)
        actions = QHBoxLayout()
        self._run = _button("batchRun", "enlarge.batch.run", self._run_batch)
        self._stop = _button("batchCancel", "enlarge.batch.cancel", self._stop_batch)
        self._stop.setEnabled(False)
        actions.addWidget(self._run, 1)
        actions.addWidget(self._stop)
        root.addLayout(actions)
        self._status = QLabel(tr("enlarge.batch.engine"))
        self._status.setObjectName("batchStatus")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        self._log = QPlainTextEdit()
        self._log.setObjectName("batchLog")
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        root.addWidget(self._log)
        self._sync_subfolders()
        self._subfolders.toggled.connect(self._sync_subfolders)
        self._restore()

    def add_paths(self, paths: list[Path]) -> None:
        """Add files or folders. Subfolders follow the shared switch."""
        found, warnings = collect(paths, recursive=self._subfolders.isChecked())
        have = {self._key(Path(str(row["src"]))) for row in self._rows}
        for path in found:
            key = self._key(path)
            if key in have:
                continue
            self._rows.append({
                "id": "",
                "src": path,
                "dst": "",
                "status": "Waiting",
                "message": "",
            })
            have.add(key)
        self._status.setText(" ".join(warnings[:4]))
        self._fill()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt name
        """Accept a drag that contains files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt name
        """Add every dropped file or folder."""
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.toLocalFile()]
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()

    def _choose_files(self) -> None:
        names, _selected = QFileDialog.getOpenFileNames(self, tr("enlarge.batch.add"), str(Path.home()), _IMAGE_FILTER)
        if names:
            self.add_paths([Path(name) for name in names])

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("enlarge.batch.folder"), str(Path.home()))
        if folder:
            self.add_paths([Path(folder)])

    def _remove(self) -> None:
        indexes = sorted({item.row() for item in self._table.selectedIndexes()}, reverse=True)
        for index in indexes:
            if 0 <= index < len(self._rows):
                del self._rows[index]
        self._fill()

    def _clear(self) -> None:
        self._rows.clear()
        self._compare.set_pair(None, None)
        self._fill()

    def _sync_subfolders(self) -> None:
        self._host.set_subfolders(self._subfolders.isChecked())

    def _key(self, path: Path) -> str:
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    def _fill(self) -> None:
        self._table.setRowCount(len(self._rows))
        for index, row in enumerate(self._rows):
            status = str(row["status"])
            label = tr(_STATUS_KEYS.get(status, "enlarge.batch.waiting"))
            if status.endswith("%"):
                label = status
            self._table.setItem(index, 0, QTableWidgetItem(Path(str(row["src"])).name))
            self._table.setItem(index, 1, QTableWidgetItem(label))
            dst = str(row["dst"])
            self._table.setItem(index, 2, QTableWidgetItem(Path(dst).name if dst else ""))
        count = len(self._rows)
        self._count.setText(tr("enlarge.batch.empty") if count == 0 else tr("enlarge.batch.count", count=count))
        self._run.setEnabled(count > 0 and not self._running)

    def _restore(self) -> None:
        saved = load_queue()
        self._rows = [
            {
                "id": str(row["id"]),
                "src": Path(str(row["src"])),
                "dst": Path(str(row["dst"])),
                "status": str(row["status"]),
                "message": str(row["message"]),
            }
            for row in saved
            if Path(str(row["src"])).is_file()
        ]
        self._fill()
        if any(still_open(row) for row in self._rows):
            self._status.setText(tr("enlarge.batch.restored"))

    def _save(self) -> None:
        if self._rows:
            dump_queue(self._rows)

    def _run_batch(self) -> None:
        if not self._host.models_ready():
            self._status.setText(tr("enlarge.error.models"))
            return
        settings = self._host.job_settings()
        pending = [row for row in self._rows if still_open(row)]
        if not pending:
            self._status.setText(tr("enlarge.batch.done"))
            return
        sources = [Path(str(row["src"])) for row in pending]
        try:
            built = requests_for(sources, settings, APP_DISPLAY_NAME)
        except (OSError, ValueError) as exc:
            self._status.setText(str(exc))
            return
        self._scales = {request.id: scale for request, scale in built}
        for row, (request, _scale) in zip(pending, built, strict=True):
            row["id"] = request.id
            row["dst"] = Path(request.dst)
            row["status"] = "Waiting"
            row["message"] = ""
        note = optional_note(settings.face, settings.generative)
        self._status.setText(note or tr("enlarge.batch.running"))
        self._cancel = False
        self._running = True
        self._run.setEnabled(False)
        self._stop.setEnabled(True)
        self._fill()
        self._save()
        started = self._host.launch(
            [request for request, _scale in built],
            lambda: self._cancel,
            self._on_progress,
            self._on_item,
        )
        if not started:
            self._running = False
            self._stop.setEnabled(False)
            self._run.setEnabled(True)
            self._status.setText(tr("enlarge.batch.busy"))

    def _stop_batch(self) -> None:
        self._cancel = True
        self._stop.setEnabled(False)
        self._status.setText(tr("enlarge.batch.stopping"))

    def _on_progress(self, job_id: str, fraction: float, message: str) -> None:
        for row in self._rows:
            if str(row["id"]) != job_id or str(row["status"]) in {"Done", "Failed", "Cancelled"}:
                continue
            row["status"] = f"{int(round(fraction * 100))}%"
            row["message"] = message
            self._fill()
            break
        self._status.setText(message)
        self._write_progress(job_id, fraction, message)

    def _on_item(self, job_id: str, ok: bool, message: str, dst: str) -> None:
        settings = self._host.job_settings()
        if ok and dst and settings.long_side > 0:
            source = next((Path(str(row["src"])) for row in self._rows if str(row["id"]) == job_id), None)
            if source is not None:
                try:
                    fit_long_side(
                        source,
                        Path(dst),
                        self._scales.get(job_id, 4),
                        settings.long_side,
                        settings.image_format,
                        settings.quality,
                        settings.lossless and settings.image_format == "jxl",
                    )
                except (OSError, ValueError) as exc:
                    ok = False
                    message = str(exc)
        for index, row in enumerate(self._rows):
            if str(row["id"]) != job_id:
                continue
            if message == "cancelled":
                row["status"] = "Cancelled"
            else:
                row["status"] = "Done" if ok else "Failed"
            row["message"] = message
            if dst:
                row["dst"] = Path(dst)
                if ok:
                    self._host.remember(job_id, Path(dst))
            self._fill()
            if ok:
                self._table.selectRow(index)
            break
        self._save()
        if all(str(row["status"]) in {"Done", "Failed", "Cancelled"} for row in self._rows):
            self._running = False
            self._stop.setEnabled(False)
            self._run.setEnabled(bool(self._rows))

    def _show_row(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if row < 0 or row >= len(self._rows):
            return
        item = self._rows[row]
        src = Path(str(item["src"]))
        dst = Path(str(item["dst"])) if item["dst"] else None
        after = dst if item["status"] == "Done" and dst is not None and dst.is_file() else None
        self._compare.set_pair(src, after)

    def _write_progress(self, job_id: str, fraction: float, message: str) -> None:
        if not self._host.job_settings().json_progress:
            return
        line = progress_line(job_id, fraction, message)
        self._log.appendPlainText(line)
        write_progress(line)


class CompareScreen(QDialog):
    """Before and after, with the same zoom controls as the old Run step."""

    def __init__(self, parent: QWidget, before: Path | None, after: Path | None) -> None:
        """Show this pair. Missing files leave that side empty."""
        super().__init__(parent)
        root = _screen(self, "compareScreen", tr("enlarge.compare.title"))
        self._view = CompareView()
        self._view.setObjectName("compareView")
        root.addWidget(self._view, 1)
        row = QHBoxLayout()
        row.addWidget(_button("compareOut", "enlarge.compare.out", lambda: self._zoom(0.8)))
        row.addWidget(_button("compareIn", "enlarge.compare.in", lambda: self._zoom(1.25)))
        row.addWidget(_button("compareFit", "enlarge.compare.fit", self._fit))
        self._zoom_label = QLabel("100%")
        self._zoom_label.setObjectName("compareZoom")
        row.addWidget(self._zoom_label)
        row.addStretch(1)
        root.addLayout(row)
        self._note = QLabel("" if before is not None and before.is_file() else tr("enlarge.compare.empty"))
        self._note.setWordWrap(True)
        root.addWidget(self._note)
        self._view.set_pair(before if before is not None and before.is_file() else None, after)

    def _zoom(self, factor: float) -> None:
        self._view.zoom_by(factor)
        self._zoom_label.setText(self._view.zoom_text())

    def _fit(self) -> None:
        self._view.zoom_fit()
        self._zoom_label.setText(self._view.zoom_text())


class CreditsScreen(QDialog):
    """Credits read from the registry, in the active language."""

    def __init__(self, parent: QWidget, models: list[ModelManifest], warnings: list[str]) -> None:
        """List every registered model and any custom-folder warning."""
        super().__init__(parent)
        root = _screen(self, "creditsScreen", tr("enlarge.credits.title"))
        sub = QLabel(f"{status_label()}  ·  {version_label()}  ·  {tr('enlarge.credits.images')}")
        sub.setObjectName("creditsMeta")
        root.addWidget(sub)
        body = QLabel(self._html(models, warnings))
        body.setObjectName("creditsBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        body.setOpenExternalLinks(True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        root.addWidget(_button("creditsClose", "enlarge.credits.close", self.accept))

    def _html(self, models: list[ModelManifest], warnings: list[str]) -> str:
        parts = [f"<p>{tr('enlarge.credits.about')}</p>", f"<p>{tr('enlarge.credits.registry')}</p>"]
        sections = credit_sections(models)
        if not sections:
            parts.append(f"<p>{tr('enlarge.credits.none')}</p>")
        for licence, group in sections:
            heading = tr("enlarge.credits.ccby") if licence.startswith("CC-BY") else licence
            parts.append(f"<p><b>{_esc(heading)}</b></p>")
            for model in group:
                parts.append(
                    "<p><b>" + _esc(model.name) + "</b><br>"
                    + _esc(model.attribution) + "<br>"
                    + _esc(model.author) + " · " + _esc(model.licence) + "<br>"
                    + _esc(model.source_url) + "</p>"
                )
        if warnings:
            parts.append(f"<p><b>{tr('enlarge.credits.custom')}</b></p>")
            parts.extend(f"<p>{_esc(warning)}</p>" for warning in warnings)
        parts.append(f"<p>{tr('enlarge.credits.other')}</p>")
        return "\n".join(parts)


class HandoffScreen(QDialog):
    """Lucent and Canvas reach this window through protocol 1."""

    def __init__(self, parent: QWidget) -> None:
        """Explain the hand-off and offer to listen, or to start a hidden listener."""
        super().__init__(parent)
        self._server: object | None = None
        root = _screen(self, "handoffScreen", tr("enlarge.handoff.title"))
        body = QLabel(tr("enlarge.handoff.body", endpoint=listen_name()))
        body.setObjectName("handoffBody")
        body.setWordWrap(True)
        root.addWidget(body)
        canvas = QLabel(tr("enlarge.handoff.canvas"))
        canvas.setWordWrap(True)
        root.addWidget(canvas)
        self._status = QLabel("")
        self._status.setObjectName("handoffStatus")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        row = QHBoxLayout()
        row.addWidget(_button("handoffListen", "enlarge.handoff.listen", self._listen))
        row.addWidget(_button("handoffHidden", "enlarge.handoff.hidden", self._hidden))
        row.addStretch(1)
        root.addLayout(row)
        root.addStretch(1)

    def _listen(self) -> None:
        # server imports this package's window. The import stays in the call.
        from enlarge.ipc.server import IpcServer

        if self._server is not None:
            self._status.setText(tr("enlarge.handoff.listening", endpoint=listen_name()))
            return
        server = IpcServer()
        server.window = self.parentWidget()  # type: ignore[assignment]
        if not server.listen():
            self._status.setText(tr("enlarge.handoff.failed"))
            return
        self._server = server
        self._status.setText(tr("enlarge.handoff.listening", endpoint=server.endpoint()))

    def _hidden(self) -> None:
        main_py = Path(__file__).resolve().parents[2] / "main.py"
        QProcess.startDetached(sys.executable, [str(main_py), "--serve", "--hidden"])
        self._status.setText(tr("enlarge.handoff.hidden_started"))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt name
        """Leave the listener running. The window owns the process."""
        super().closeEvent(event)


class SetupScreen(QDialog):
    """Process commands from the CLI, each as its own action."""

    def __init__(self, parent: QWidget) -> None:
        """List version, capabilities, register, install, and the worker note."""
        super().__init__(parent)
        self._thread: _SetupThread | None = None
        root = _screen(self, "setupScreen", tr("enlarge.setup.title"))
        self._exe = QLineEdit(str(Path(sys.argv[0]).resolve()) if sys.argv else "")
        self._exe.setObjectName("setupExe")
        self._exe.setPlaceholderText(tr("enlarge.setup.exe"))
        root.addWidget(self._exe)
        self._model = QLineEdit()
        self._model.setObjectName("setupModel")
        self._model.setPlaceholderText(tr("enlarge.setup.prepare"))
        root.addWidget(self._model)
        self._json = QCheckBox(tr("enlarge.setup.json"))
        self._json.setObjectName("setupJson")
        self._json.setChecked(True)
        root.addWidget(self._json)
        grid = QHBoxLayout()
        column = QVBoxLayout()
        for name, key, command in (
            ("setupVersion", "enlarge.setup.version", "version"),
            ("setupCapabilities", "enlarge.setup.capabilities", "capabilities"),
            ("setupRegister", "enlarge.setup.register", "register"),
            ("setupUnregister", "enlarge.setup.unregister", "unregister"),
            ("setupSelftest", "enlarge.setup.selftest", "selftest"),
            ("setupInstallGpu", "enlarge.setup.install_gpu", "install-gpu"),
            ("setupPrepare", "enlarge.setup.prepare_button", "prepare-model"),
            ("setupInstallSuite", "enlarge.setup.install_suite", "install-suite"),
            ("setupUninstall", "enlarge.setup.uninstall", "uninstall-suite"),
            ("setupHidden", "enlarge.setup.hidden", "hidden"),
            ("setupWorker", "enlarge.setup.worker", "worker"),
        ):
            column.addWidget(_button(name, key, self._bind(command)))
        column.addStretch(1)
        grid.addLayout(column)
        self._result = QPlainTextEdit()
        self._result.setObjectName("setupResult")
        self._result.setReadOnly(True)
        grid.addWidget(self._result, 1)
        root.addLayout(grid, 1)

    def _bind(self, command: str) -> Callable[[], None]:
        """Return a click handler for this setup command."""

        def run() -> None:
            self._run(command)

        return run

    def _run(self, command: str) -> None:
        if command == "worker":
            self._result.setPlainText(tr("enlarge.setup.worker_reason"))
            return
        if command == "uninstall-suite":
            answer = QMessageBox.question(self, APP_DISPLAY_NAME, tr("enlarge.setup.uninstall_ask"))
            if answer != QMessageBox.StandardButton.Yes:
                return
        if command == "hidden":
            main_py = Path(__file__).resolve().parents[2] / "main.py"
            QProcess.startDetached(sys.executable, [str(main_py), "--serve", "--hidden"])
            self._result.setPlainText(tr("enlarge.handoff.hidden_started"))
            return
        self._thread = _SetupThread(command, self._exe.text().strip(), self._model.text().strip(), self._json.isChecked())
        self._thread.reported.connect(self._result.setPlainText)
        self._thread.start()


class DryScreen(QDialog):
    """The plan for the current picture. Nothing is written."""

    def __init__(self, parent: QWidget, lines: list[str], command: str) -> None:
        """Show the planned paths and the matching command."""
        super().__init__(parent)
        root = _screen(self, "dryScreen", tr("enlarge.dry.title"))
        text = QPlainTextEdit()
        text.setObjectName("dryBody")
        text.setReadOnly(True)
        text.setPlainText("\n".join([*lines, "", command]))
        root.addWidget(text, 1)


class _SetupThread(QThread):
    """Run one setup command off the widget thread."""

    reported = Signal(str)

    def __init__(self, command: str, exe: str, model_id: str, as_json: bool) -> None:
        """Remember which command to run."""
        super().__init__()
        self._command = command
        self._exe = exe
        self._model_id = model_id
        self._json = as_json

    def run(self) -> None:
        """Capture the command's text and emit it."""
        try:
            self.reported.emit(setup_command(self._command, self._exe, self._model_id, self._json))
        except (OSError, RuntimeError, ValueError) as exc:
            self.reported.emit(str(exc))


def write_progress(line: str) -> None:
    """Append one JSON progress line under the Enlarge state directory."""
    path = state_dir() / "enlarge" / "progress.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        return


def setup_command(command: str, exe: str, model_id: str, as_json: bool) -> str:
    """Run one CLI command and return what it printed."""
    buffer = io.StringIO()
    errors = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
        if command == "version":
            buffer.write(version_label())
        elif command == "capabilities":
            if not as_json:
                errors.write("usage: voxprint-enlarge --capabilities --json\n")
                code = 2
            else:
                buffer.write(json.dumps(capabilities(), indent=2))
        elif command == "register":
            register_install(Path(exe or str(Path(__file__).resolve().parents[2] / "main.py")))
            buffer.write("registered\n")
        elif command == "unregister":
            unregister_install()
            buffer.write("unregistered\n")
        elif command == "selftest":
            code = entry._selftest(as_json)
        elif command == "install-gpu":
            code = entry._install_gpu()
        elif command == "prepare-model":
            code = entry._prepare_model(model_id)
        elif command == "install-suite":
            code = entry._install_suite()
        elif command == "uninstall-suite":
            code = entry._uninstall_suite()
        else:
            errors.write(str(command))
            code = 2
    text = buffer.getvalue().strip()
    error = errors.getvalue().strip()
    parts = [str(code)]
    if text:
        parts.append(text)
    if error:
        parts.append(error)
    return "\n".join(parts)


def show_hash(parent: QWidget, model_id: str) -> None:
    """Compare the selected weight file with its pinned sha256."""
    model = load_builtin(release=False).get(model_id)
    if model is None:
        QMessageBox.information(parent, APP_DISPLAY_NAME, tr("enlarge.hash.missing"))
        return
    state, detail = weight_report(model)
    QMessageBox.information(parent, tr("enlarge.hash.title"), f"{model.filename}\n{state}\n{detail}")


def remove_weights(parent: QWidget, model_id: str) -> None:
    """Delete the selected weight when no other app still uses it."""
    model = load_builtin(release=False).get(model_id)
    if model is None or not model.filename:
        QMessageBox.information(parent, APP_DISPLAY_NAME, tr("enlarge.hash.missing"))
        return
    answer = QMessageBox.question(parent, APP_DISPLAY_NAME, tr("enlarge.weights.ask", name=model.filename))
    if answer != QMessageBox.StandardButton.Yes:
        return
    try:
        delete_weight(model)
    except (OSError, StoreError) as exc:
        QMessageBox.warning(parent, APP_DISPLAY_NAME, str(exc))
        return
    QMessageBox.information(parent, APP_DISPLAY_NAME, tr("enlarge.weights.done"))


def show_dry_run(parent: QWidget, source: Path, settings: JobSettings) -> None:
    """Open the plan for this picture."""
    try:
        lines = describe_job(source, settings) if source.is_file() else [tr("enlarge.error.missing")]
    except (OSError, ValueError) as exc:
        lines = [str(exc)]
    DryScreen(parent, lines, command_line(source, settings)).exec()


def show_credits(parent: QWidget, custom_models: str) -> CreditsScreen:
    """Open credits for the built-in registry plus a custom folder."""
    folder = Path(custom_models) if custom_models.strip() else None
    catalog = combined_catalog(folder, release=True)
    dialog = CreditsScreen(parent, list(catalog.models), list(catalog.warnings))
    dialog.show()
    return dialog


def optional_note(face: str, generative: str) -> str:
    """Sentence shown when a registered step is stored and this build does not run it."""
    parts: list[str] = []
    if face and face != "off":
        parts.append(tr("enlarge.note.face"))
    if generative and generative != "off":
        parts.append(tr("enlarge.note.generative"))
    return " ".join(parts)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
