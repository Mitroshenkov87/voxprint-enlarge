"""Convert a ``.pth`` to ONNX fp16 with spandrel's MIT core.

The loader is the ``spandrel`` package. ``spandrel_extra_arches`` is not
imported; an architecture that exists only there fails to load.

Usage:
  python tools/export_onnx.py WEIGHT.pth [--out WEIGHT.onnx]

The ``.onnx`` is written next to the ``.pth`` unless ``--out`` is set. Put that
file in the shared models folder beside the weight. The engine prefers it over
the PyTorch fallback.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_torch: Any = None
_loader: Any = None
try:
    import torch as _torch_mod
    from spandrel import ModelLoader as _loader_cls

    _torch = _torch_mod
    _loader = _loader_cls
except ImportError:
    pass

_onnx: Any = None
_to_fp16: Any = None
try:
    import onnx as _onnx_mod
    from onnxruntime.transformers.float16 import convert_float_to_float16 as _fp16

    _onnx = _onnx_mod
    _to_fp16 = _fp16
except ImportError:
    pass


def convert(weight: Path, dest: Path) -> None:
    """Load ``weight`` on CPU and write an fp16 ONNX file to ``dest``."""
    if _torch is None or _loader is None or _onnx is None or _to_fp16 is None:
        raise SystemExit("export needs torch, spandrel, onnx, and onnxruntime. See docs/ENGINE.md.")
    descriptor = _loader().load_from_file(str(weight))
    module = descriptor.model
    module.eval()
    module = module.cpu().float()
    example = _torch.rand(1, 3, 64, 64)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _torch.onnx.export(
        module,
        example,
        str(dest),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height_out", 3: "width_out"},
        },
        opset_version=17,
    )
    graph = _onnx.load(str(dest))
    _onnx.save(_to_fp16(graph, keep_io_types=False), str(dest))


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
    convert(weight, dest)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
