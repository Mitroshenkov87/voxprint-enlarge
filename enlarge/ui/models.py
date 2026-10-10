"""Background download of one pinned weight. The window thread only receives signals."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.store import StoreError, ensure_weight


class DownloadThread(QThread):
    progressed = Signal(float, str)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, model: ModelManifest, root: Path) -> None:
        super().__init__()
        self.model = model
        self.root = root

    def run(self) -> None:
        try:
            path = ensure_weight(self.model, self.root, on_progress=self.progressed.emit)
        except (OSError, StoreError) as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(str(path))
