"""PyTorch fallback used only when the ONNX session cannot be built.

The loader is ``spandrel`` itself (MIT). ``spandrel_extra_arches`` is never
imported. A model whose architecture is not in the MIT core fails here instead
of pulling in the extra package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from spandrel import ModelLoader
except ImportError:  # torch and spandrel are optional; the ONNX path does not need them
    torch = None
    ModelLoader = None


class TorchFallbackError(RuntimeError):
    """The .pth file could not be run. The message says whether the package is missing."""


def torch_available() -> bool:
    """True when PyTorch imports in this process."""
    return torch is not None and ModelLoader is not None


class TorchUpscaler:
    """One loaded .pth. fp16 on CUDA, fp32 otherwise."""

    def __init__(self, weight: Path, device: str) -> None:
        """Load a spandrel weight onto this device."""
        if torch is None or ModelLoader is None:
            raise TorchFallbackError("PyTorch fallback needs torch and spandrel (the MIT package, not spandrel_extra_arches)")
        if "spandrel_extra_arches" in sys.modules:
            raise TorchFallbackError("spandrel_extra_arches must not be loaded")
        use_cuda = torch.cuda.is_available() and not device.strip().lower().startswith("cpu")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.fp16 = use_cuda
        try:
            descriptor = ModelLoader().load_from_file(str(weight))
        except (OSError, RuntimeError, ValueError) as exc:
            raise TorchFallbackError(f"could not load {weight.name}: {exc}") from exc
        module = descriptor.model
        module.eval()
        if self.fp16:
            module = module.half()
        self.module = module.to(self.device)
        self.scale = int(getattr(descriptor, "scale", 0) or 0)

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        """Run the network on one NCHW tensor and return the image."""
        if torch is None:
            raise TorchFallbackError("torch is not installed")
        array = np.transpose(tile, (2, 0, 1))[None, ...]
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        dtype = torch.float16 if self.fp16 else torch.float32
        tensor = tensor.to(device=self.device, dtype=dtype)
        with torch.inference_mode():
            output = self.module(tensor)
        image = output.float().clamp(0, 1).cpu().numpy()[0]
        return np.transpose(image, (1, 2, 0))
