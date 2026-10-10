"""Background queue. The widget thread only receives signals."""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QThread, Signal

from enlarge.batch import run_batch
from voxprint_engine.worker import UpscaleRequest


class BatchThread(QThread):
    progressed = Signal(str, float, str)
    item_finished = Signal(str, bool, str, str)

    def __init__(self, requests: list[UpscaleRequest], cancel: Callable[[], bool]) -> None:
        super().__init__()
        self.requests = requests
        self._cancel = cancel

    def run(self) -> None:
        run_batch(
            self.requests,
            on_progress=self.progressed.emit,
            on_item=self.item_finished.emit,
            cancel=self._cancel,
        )
