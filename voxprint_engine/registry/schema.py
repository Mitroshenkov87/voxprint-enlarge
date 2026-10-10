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

    def requires_credit(self) -> bool:
        """CC-BY (any version) needs a line on the credits screen. NC never reaches here."""
        return self.licence.startswith("CC-BY")
