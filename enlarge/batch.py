"""Run a queue through the worker process. Qt is not imported here."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from voxprint_engine.worker import UpscaleRequest, WorkerClient, WorkerError
from voxprint_engine.worker.protocol import UpscaleResult


class BatchControl:
    """Cancel flag the queue and a background click can share."""

    def __init__(self) -> None:
        self.cancelled = False
        self.client: WorkerClient | None = None

    def cancel(self) -> None:
        self.cancelled = True
        client = self.client
        if client is not None:
            client.interrupt()

    def is_cancelled(self) -> bool:
        return self.cancelled


def run_batch(
    requests: list[UpscaleRequest],
    on_progress: Callable[[str, float, str], None],
    on_item: Callable[[str, bool, str, str], None],
    cancel: Callable[[], bool],
    control: BatchControl | None = None,
) -> list[UpscaleResult]:
    """Upscale each request. A failed item is reported and the queue continues.

    ``next_batch`` from the worker is the suite spill rule. Enlarge sends one
    image at a time, so the number stays 1 unless the worker reports a spill.
    A cancel stops the worker that is running and leaves the remaining images
    unstarted.
    """
    done: list[UpscaleResult] = []
    if not requests:
        return done
    held = control or BatchControl()
    batch = max(1, requests[0].batch)
    with WorkerClient() as client:
        held.client = client
        try:
            for request in requests:
                if cancel() or held.is_cancelled():
                    on_item(request.id, False, "cancelled", "")
                    break
                current = replace(request, batch=batch)
                try:

                    def _progress(fraction: float, message: str, job_id: str = current.id) -> None:
                        on_progress(job_id, fraction, message)
                        if cancel() or held.is_cancelled():
                            client.interrupt()

                    result = client.upscale(current, on_progress=_progress)
                except WorkerError as exc:
                    if cancel() or held.is_cancelled():
                        on_item(current.id, False, "cancelled", "")
                        break
                    detail = str(exc)
                    extra = client.stderr_text().strip()
                    if extra and extra not in detail:
                        detail = f"{detail} ({extra})"
                    on_item(current.id, False, detail, "")
                    continue
                batch = max(1, result.next_batch)
                done.append(result)
                on_item(result.id, True, result.message, result.dst)
        finally:
            held.client = None
    return done
