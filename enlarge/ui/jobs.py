"""Background queue. The widget thread only receives signals."""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QThread, Signal

from enlarge.batch import BatchControl, run_batch
from voxprint_engine.worker import UpscaleRequest


class BatchThread(QThread):
    """Background thread that runs one batch and emits progress."""
    progressed = Signal(str, float, str)
    item_finished = Signal(str, bool, str, str)

    def __init__(self, requests: list[UpscaleRequest], cancel: Callable[[], bool]) -> None:
        """Hold the requests and the shared cancel flag."""
        super().__init__()
        self.requests = requests
        self._cancel = cancel
        self.control = BatchControl()

    def cancel(self) -> None:
        """Ask the batch to stop."""
        self.control.cancel()

    def run(self) -> None:
        """Run the batch and emit item and finished signals."""
        run_batch(
            self.requests,
            on_progress=self.progressed.emit,
            on_item=self.item_finished.emit,
            cancel=self._cancel,
            control=self.control,
        )


class ModelInstallThread(QThread):
    """Download and prepare one model off the widget thread."""
    progressed = Signal(str)
    done = Signal(bool, str)

    def __init__(self, model_id: str) -> None:
        """Hold the model id."""
        super().__init__()
        self.model_id = model_id

    def run(self) -> None:
        """Run ``install_model`` and emit ``done`` with the outcome."""
        from enlarge.suite_runtime import SuiteInstallError, install_model

        try:
            install_model(self.model_id, self.progressed.emit)
        except SuiteInstallError as exc:
            self.done.emit(False, str(exc))
            return
        self.done.emit(True, "")
