"""Background download and ONNX conversion. The window thread only receives signals."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from voxprint_engine.onnx_export import OnnxExportError, ensure_onnx
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.store import StoreError


class DownloadThread(QThread):
    """Download one pinned weight and convert it to a cached ONNX file."""
    progressed = Signal(float, str)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, model: ModelManifest, root: Path) -> None:
        """Hold the model and the models directory."""
        super().__init__()
        self.model = model
        self.root = root

    def run(self) -> None:
        """Download, convert, and emit progress, then the ONNX path or the error."""
        try:
            path = ensure_onnx(self.model, self.root, on_progress=self.progressed.emit)
        except (OSError, StoreError, OnnxExportError) as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(str(path))
