"""Run a queue through the worker process. Qt is not imported here."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from voxprint_engine.worker import UpscaleRequest, WorkerClient, WorkerError
from voxprint_engine.worker.protocol import UpscaleResult


def run_batch(
    requests: list[UpscaleRequest],
    on_progress: Callable[[str, float, str], None],
    on_item: Callable[[str, bool, str, str], None],
    cancel: Callable[[], bool],
) -> list[UpscaleResult]:
    """Upscale each request. A failed item is reported and the queue continues.

    ``next_batch`` from the worker is the suite spill rule. Enlarge sends one
    image at a time, so the number stays 1 unless the worker reports a spill.
    """
    done: list[UpscaleResult] = []
    if not requests:
        return done
    batch = max(1, requests[0].batch)
    with WorkerClient() as client:
        for request in requests:
            if cancel():
                on_item(request.id, False, "cancelled", "")
                break
            current = replace(request, batch=batch)
            try:
                def _progress(fraction: float, message: str, job_id: str = current.id) -> None:
                    on_progress(job_id, fraction, message)

                result = client.upscale(current, on_progress=_progress)
            except WorkerError as exc:
                detail = str(exc)
                extra = client.stderr_text().strip()
                if extra and extra not in detail:
                    detail = f"{detail} ({extra})"
                on_item(current.id, False, detail, "")
                continue
            batch = max(1, result.next_batch)
            done.append(result)
            on_item(result.id, True, result.message, result.dst)
    return done
