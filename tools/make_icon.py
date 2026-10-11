"""Scale assets/enlarge.png into the icon set. Does not redraw the mark or the splash.

    python tools/make_icon.py

The 1024 PNG is the supplied artwork. This script never writes it back.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _frame(size: int, source: Image.Image) -> Image.Image:
    return source.resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")


def write_icons() -> None:
    source_path = ASSETS / "enlarge.png"
    source = Image.open(source_path)
    if source.size != (1024, 1024):
        raise SystemExit(f"{source_path} must be 1024x1024, got {source.size}")
    rgba = source.convert("RGBA")
    frames = [_frame(size, rgba) for size in ICO_SIZES]
    linux = ROOT / "installer" / "linux"
    linux.mkdir(parents=True, exist_ok=True)
    frames[-1].save(linux / "voxprint-enlarge-256.png", optimize=True)
    frames[-1].save(ASSETS / "enlarge-256.png", optimize=True)
    # Pillow wants the largest frame first when saving a multi-size ICO.
    frames[-1].save(
        ASSETS / "enlarge.ico",
        format="ICO",
        sizes=[(size, size) for size in ICO_SIZES],
        append_images=frames[:-1],
    )


if __name__ == "__main__":
    write_icons()
    print("wrote", ASSETS / "enlarge.ico")
