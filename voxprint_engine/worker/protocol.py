"""JSON-line messages between the app and the worker process.

One JSON object per line. The child writes logs to stderr so stdout stays a
protocol stream. ``op`` is ``ready``, ``upscale``, ``restore_batch``,
``progress``, ``result``, ``error``, ``ping``, or ``shutdown``.
``RestoreBatchRequest`` is the video op added for Clarity. API stays 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from voxprint_engine.video import RestoreBatchRequest

__all__ = ["UpscaleRequest", "UpscaleResult", "RestoreBatchRequest"]


@dataclass(frozen=True)
class UpscaleRequest:
    """One image job sent to the worker."""
    id: str
    src: str
    dst: str
    scale: int
    format: str
    quality: int
    model_id: str
    tile: int = 0
    model_tile: int = 0
    double: bool = False
    batch: int = 1
    device: str = "auto"
    engine: str = "auto"
    custom_models_dir: str = ""
    lossless: bool = False
    app_name: str = ""

    def to_message(self) -> dict[str, Any]:
        """The upscale JSON line for this request."""
        return {
            "op": "upscale",
            "id": self.id,
            "src": self.src,
            "dst": self.dst,
            "scale": self.scale,
            "format": self.format,
            "quality": self.quality,
            "lossless": self.lossless,
            "model_id": self.model_id,
            "tile": self.tile,
            "model_tile": self.model_tile,
            "double": self.double,
            "batch": self.batch,
            "device": self.device,
            "engine": self.engine,
            "custom_models_dir": self.custom_models_dir,
            "app_name": self.app_name,
        }


@dataclass(frozen=True)
class UpscaleResult:
    """The worker's result line, including tile and spill fields."""
    id: str
    ok: bool
    message: str
    dst: str = ""
    width: int = 0
    height: int = 0
    scale: int = 0
    engine: str = ""
    model_id: str = ""
    tile: int = 0
    overlap: int = 0
    vram_budget_gb: float = 0.0
    spilled: bool = False
    next_batch: int = 1

    @classmethod
    def from_message(cls, message: Mapping[str, Any]) -> UpscaleResult:
        """Build a result from a worker JSON object."""
        ok = bool(message.get("ok"))
        return cls(
            id=str(message.get("id") or ""),
            ok=ok,
            message=str(message.get("message") or ("ok" if ok else "failed")),
            dst=str(message.get("dst") or ""),
            width=int(message.get("width") or 0),
            height=int(message.get("height") or 0),
            scale=int(message.get("scale") or 0),
            engine=str(message.get("engine") or ""),
            model_id=str(message.get("model_id") or ""),
            tile=int(message.get("tile") or 0),
            overlap=int(message.get("overlap") or 0),
            vram_budget_gb=float(message.get("vram_budget_gb") or 0.0),
            spilled=bool(message.get("spilled")),
            next_batch=int(message.get("next_batch") or 1),
        )
