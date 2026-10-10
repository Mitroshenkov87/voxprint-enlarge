"""Overlapping tiles and a separable feather.

The weight on an interior edge ramps across ``overlap`` pixels. The two tiles
that share that edge add up to 1. A tile that sits on the picture border does
not fade on that border, so the rim is not dimmed. Scale is applied after the
network: overlap in the output is ``overlap * scale``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class TileBox:
    x: int
    y: int
    width: int
    height: int


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    last = length - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def plan_tiles(width: int, height: int, tile: int, overlap: int) -> list[TileBox]:
    """Cover ``width`` by ``height``. The last tile is pulled back so it stays inside."""
    if width < 1 or height < 1:
        raise ValueError("the image is empty")
    if tile < 1:
        raise ValueError("tile must be positive")
    overlap = max(0, min(int(overlap), int(tile) // 2))
    stride = max(1, int(tile) - overlap)
    boxes: list[TileBox] = []
    for y in _starts(height, tile, stride):
        for x in _starts(width, tile, stride):
            boxes.append(TileBox(x, y, min(tile, width - x), min(tile, height - y)))
    return boxes


def feather(height: int, width: int, overlap: int, *, fade_left: bool, fade_right: bool, fade_top: bool, fade_bottom: bool) -> Array:
    """HW weight in 0..1. Interior edges ramp; border edges stay at 1."""
    weight_y = np.ones(height, dtype=np.float32)
    weight_x = np.ones(width, dtype=np.float32)
    if overlap > 0:
        ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / float(overlap)
        if fade_top and overlap <= height:
            weight_y[:overlap] *= ramp
        if fade_bottom and overlap <= height:
            weight_y[-overlap:] *= ramp[::-1]
        if fade_left and overlap <= width:
            weight_x[:overlap] *= ramp
        if fade_right and overlap <= width:
            weight_x[-overlap:] *= ramp[::-1]
    return weight_y[:, None] * weight_x[None, :]


def blend(height: int, width: int, channels: int, pieces: list[tuple[int, int, Array, Array]]) -> Array:
    """Average ``pieces`` of ``(y, x, hwc float, hw weight)`` into one image."""
    acc = np.zeros((height, width, channels), dtype=np.float32)
    cover = np.zeros((height, width, 1), dtype=np.float32)
    for y, x, image, weight in pieces:
        tile_h, tile_w = image.shape[0], image.shape[1]
        acc[y:y + tile_h, x:x + tile_w] += image * weight[..., None]
        cover[y:y + tile_h, x:x + tile_w] += weight[..., None]
    return acc / np.maximum(cover, 1e-6)


def upscale_tiled(
    image: Array,
    infer: Callable[[Array], Array],
    tile: int,
    overlap: int,
    scale: int,
) -> Array:
    """Run ``infer`` on each tile. ``infer`` receives and returns HWC float32 in 0..1.

    The returned tile must be exactly ``scale`` times the input tile. Tiles
    smaller than ``tile`` (the right and bottom edges) are padded by edge
    replication and the padding is cropped off the network output.
    """
    if image.ndim != 3:
        raise ValueError("expected HWC")
    if scale < 1:
        raise ValueError("scale must be >= 1")
    height, width, channels = image.shape
    boxes = plan_tiles(width, height, tile, overlap)
    out_h, out_w = height * scale, width * scale
    pieces: list[tuple[int, int, Array, Array]] = []
    for box in boxes:
        crop = image[box.y:box.y + box.height, box.x:box.x + box.width]
        # Edge tiles are shorter than ``tile``. Pad so a fixed-size network still runs,
        # then crop the padding back off the output.
        pad_h = max(0, tile - box.height)
        pad_w = max(0, tile - box.width)
        if pad_h > 0 or pad_w > 0:
            crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        produced = np.asarray(infer(crop), dtype=np.float32)
        expect_h = crop.shape[0] * scale
        expect_w = crop.shape[1] * scale
        if produced.shape[0] != expect_h or produced.shape[1] != expect_w or produced.shape[2] != channels:
            raise ValueError(
                f"the network returned {produced.shape}, expected {(expect_h, expect_w, channels)}"
            )
        produced = produced[: box.height * scale, : box.width * scale]
        weight = feather(
            produced.shape[0],
            produced.shape[1],
            overlap * scale,
            fade_left=box.x > 0,
            fade_right=box.x + box.width < width,
            fade_top=box.y > 0,
            fade_bottom=box.y + box.height < height,
        )
        pieces.append((box.y * scale, box.x * scale, produced, weight))
    blended = blend(out_h, out_w, channels, pieces)
    return np.clip(blended, 0.0, 1.0)
