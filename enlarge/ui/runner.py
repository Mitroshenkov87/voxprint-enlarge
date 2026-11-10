"""Run a batch while holding the suite GPU lock the way a headless job does."""
from __future__ import annotations

import os
from collections.abc import Callable

from voxprint_common.gpu_lock import GpuLockTimeout, gpu_job

from enlarge.ui.jobs import BatchThread
from voxprint_engine.worker import UpscaleRequest

_PARENT = "VOXPRINT_GPU_PARENT_PID"


class GuidedBatchThread(BatchThread):
    """A batch that waits for the suite GPU lock, then shares that hold with the worker."""

    def __init__(
        self,
        requests: list[UpscaleRequest],
        cancel: Callable[[], bool],
        parent_pid: int,
        lock_timeout: float,
    ) -> None:
        """Keep the lock wait beside the requests.

        Args:
            requests: Images to enlarge, in order.
            cancel: Called from the worker. True asks the batch to stop.
            parent_pid: Pid that already holds the lock. Zero means this process takes it.
            lock_timeout: Seconds to wait for the lock. The CLI default is 600.
        """
        super().__init__(requests, cancel)
        self.parent_pid = parent_pid
        self.lock_timeout = lock_timeout

    def run(self) -> None:
        """Take the lock, then run the batch. A timeout reports every image as not started."""
        previous = os.environ.get(_PARENT)
        try:
            if self.parent_pid > 0:
                os.environ[_PARENT] = str(self.parent_pid)
                super().run()
                return
            os.environ[_PARENT] = str(os.getpid())
            with gpu_job(
                "enlarge",
                "gui",
                eta=max(60.0, self.lock_timeout or 60.0),
                timeout=self.lock_timeout,
                poll=0.05,
                cancel=self._cancel,
            ):
                super().run()
        except GpuLockTimeout as exc:
            text = str(exc) or "the GPU is busy"
            for request in self.requests:
                self.item_finished.emit(request.id, False, text, "")
        finally:
            if previous is None:
                os.environ.pop(_PARENT, None)
            else:
                os.environ[_PARENT] = previous
