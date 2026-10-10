"""Time one image on this machine.

CI does not run this script. The steps and the GPU packages are in docs/ENGINE.md.

  python tools/benchmark_gpu.py --image samples/sample.png --model 4x-nomos2-realplksr-dysample --scale 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from voxprint_engine.formats import DEFAULT_FORMAT, DEFAULT_QUALITY
from voxprint_engine.gpu_lock import gpu_job
from voxprint_engine.infer import preflight, upscale_file, uses_lanczos
from voxprint_engine.registry import load_builtin
from voxprint_engine.store import StoreError, ensure_weight
from voxprint_engine.worker.lanczos import ImageJobError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time one upscale on the local GPU (or CPU ONNX).")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="4x-nomos2-realplksr-dysample")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile", type=int, default=0)
    args = parser.parse_args(argv)
    catalog = load_builtin(release=True)
    model = catalog.get(args.model)
    if model is None or not model.selectable:
        print(f"unknown or unavailable model {args.model}", file=sys.stderr)
        return 2
    dest = args.out or args.image.with_name(f"{args.image.stem}_bench.avif")
    try:
        preflight(args.image, dest, args.scale, DEFAULT_FORMAT, DEFAULT_QUALITY, False)
        weight = None
        if not uses_lanczos(model, args.device, "auto"):
            weight = ensure_weight(model)
        started = time.perf_counter()
        with gpu_job(f"enlarge:bench:{model.id}", eta_s=600.0, timeout=30.0):
            outcome = upscale_file(
                args.image,
                dest,
                model,
                scale=args.scale,
                image_format=DEFAULT_FORMAT,
                quality=DEFAULT_QUALITY,
                double=False,
                device=args.device,
                user_tile=args.tile,
                batch=1,
                engine="auto",
                weight=weight,
            )
    except (ImageJobError, StoreError, TimeoutError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"{outcome.engine} {outcome.message} {outcome.width}x{outcome.height} "
        f"tile {outcome.tile} {elapsed_ms:.1f} ms -> {dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
