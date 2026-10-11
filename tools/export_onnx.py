"""Convert a ``.pth`` to ONNX fp16 with the engine exporter.

The loader is the ``spandrel`` package. ``spandrel_extra_arches`` is not
imported; an architecture that exists only there fails to load. Torch 2.14
must export with ``dynamo=False``; that is what ``export_weight`` does, and
``onnx.checker`` has to accept the graph before it is written.

Usage:
  python tools/export_onnx.py WEIGHT.pth [--out WEIGHT.onnx]

The app converts during install and during a model download, and caches the
file as ``<sha256>.onnx`` in the shared models folder. This script is the
same conversion for a file outside that folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxprint_engine.onnx_export import OnnxExportError, export_weight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a .pth (spandrel MIT core) to ONNX fp16.")
    parser.add_argument("weight", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    weight = args.weight
    if not weight.is_file():
        print(f"not a file: {weight}", file=sys.stderr)
        return 2
    dest = args.out or weight.with_suffix(".onnx")
    try:
        export_weight(weight, dest)
    except OnnxExportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
