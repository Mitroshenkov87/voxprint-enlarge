"""Model scale for a target long side, and the exact size after the downscale.

Native models are 4x. Scale 2 is that pass followed by a downscale to 2x.
``--long-side N`` picks the smallest of 2 and 4 that reaches N, then resizes
the result so the long side is exactly N. Past 4x is ``bad_input``.
"""
from __future__ import annotations

MODEL_SCALES = (2, 4)


def choose_scale(width: int, height: int, long_side: int) -> int:
    """Smallest model scale whose long side is at least ``long_side``."""
    if width < 1 or height < 1 or long_side < 1:
        raise ValueError("long side must be a positive size")
    long_edge = max(width, height)
    for scale in MODEL_SCALES:
        if scale * long_edge >= long_side:
            return scale
    raise ValueError(f"long side {long_side} is larger than a 4x enlarge of this image ({long_edge} px)")


def fit_long_side(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Resize so ``max(width, height) == long_side``. The short side uses round-half-even and is at least 1."""
    if width < 1 or height < 1 or long_side < 1:
        raise ValueError("long side must be a positive size")
    if width >= height:
        short = max(1, int(round(height * long_side / width)))
        return long_side, short
    short = max(1, int(round(width * long_side / height)))
    return short, long_side


def planned_size(width: int, height: int, scale: int, long_side: int = 0) -> tuple[int, int]:
    """Pixel size after the model scale, then the exact long-side fit when ``long_side`` is set."""
    grown_w = width * scale
    grown_h = height * scale
    if long_side:
        return fit_long_side(grown_w, grown_h, long_side)
    return grown_w, grown_h
