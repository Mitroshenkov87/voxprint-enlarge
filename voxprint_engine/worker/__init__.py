"""Inference worker: a separate process, JSON lines on stdin/stdout.

Phase 1 answers ``upscale`` with a Lanczos resize so the app flow can be tried
without weights. The process still takes the suite GPU lock and reports the
tile the real engine would use.
"""
from __future__ import annotations

from voxprint_engine.worker.client import WorkerClient, WorkerError
from voxprint_engine.worker.protocol import UpscaleRequest, UpscaleResult

__all__ = ["UpscaleRequest", "UpscaleResult", "WorkerClient", "WorkerError"]
