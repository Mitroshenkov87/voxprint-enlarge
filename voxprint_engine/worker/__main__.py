"""``python -m voxprint_engine.worker`` — the child process the app starts."""
from __future__ import annotations

from voxprint_engine.trt_libs import prepare_native_runtime

# The server import loads onnxruntime. Register CUDA 13 first.
prepare_native_runtime()

from voxprint_engine.worker.server import serve

if __name__ == "__main__":
    raise SystemExit(serve())
