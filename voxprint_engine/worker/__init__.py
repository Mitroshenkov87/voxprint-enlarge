"""Inference worker: a separate process, JSON lines on stdin/stdout.

An upscale runs ONNX on TensorRT. CUDA runs when that engine does not build.
``VOXPRINT_CI_LANCZOS=1`` resamples with Lanczos for tests and CI. The process
holds the suite GPU lock for the inference itself. Downloads happen before that
lock.
"""
from __future__ import annotations

from voxprint_engine.worker.client import WorkerClient, WorkerError
from voxprint_engine.worker.protocol import UpscaleRequest, UpscaleResult

__all__ = ["UpscaleRequest", "UpscaleResult", "WorkerClient", "WorkerError"]
