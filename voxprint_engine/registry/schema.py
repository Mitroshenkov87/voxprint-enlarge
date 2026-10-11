"""One manifest per model. Field names are the contract Clarity imports."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelManifest:
    """A licence-bearing description of one upscale or restore model.

    ``sha256`` is the lowercase hex digest of the weight file. It stays empty
    until that file is pinned; a non-empty value must already be 64 hex chars.
    ``vram_mb`` and ``tile`` are hints for the tiler, not measurements.
    ``tile`` 0 means the worker may pick a tile from the VRAM budget.
    ``onnx_url`` and ``onnx_sha256`` name a prebuilt ONNX file when a mirror
    publishes one. ``execution`` is ``auto``, ``cuda``, or ``tensorrt``.
    ``in_channels`` is the network input depth. 3 is RGB. 4 is RGB plus a
    noise-level map, which is what DRUNet takes.
    """

    id: str
    name: str
    arch: str
    scale: int
    role: str
    licence: str
    author: str
    attribution: str
    source_url: str
    sha256: str
    vram_mb: int
    tile: int
    filename: str
    summary: str
    selectable: bool
    download_url: str = ""
    onnx_url: str = ""
    onnx_sha256: str = ""
    execution: str = "auto"
    in_channels: int = 3

    def requires_credit(self) -> bool:
        """CC-BY (any version) needs a line on the credits screen. NC never reaches here."""
        return self.licence.startswith("CC-BY")


def checked_in_channels(value: object) -> int:
    """Return ``value`` when it is an integer from 1 to 16.

    Bool is rejected. The export example and the TensorRT profile both use
    this count, so a bad value would build the wrong engine.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 16:
        raise ValueError("in_channels must be an integer from 1 to 16")
    return value
