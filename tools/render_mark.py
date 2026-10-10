"""Draw assets/enlarge.png, the Enlarge mark, and a synthetic sample picture.

The mark matches the supplied app icon: a navy rounded tile, a dashed selection,
a picture glyph, and a two-way enlarge arrow, in a pink-to-coral neon. Run this
only to regenerate that source. tools/make_icon.py scales the PNG; it does not
redraw the glyph.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SAMPLES = ROOT / "samples"


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _stamp(base: Image.Image, mask: Image.Image, color: tuple[int, int, int], blur: float, gain: float) -> None:
    glow = mask.filter(ImageFilter.GaussianBlur(blur))
    layer = Image.new("RGBA", base.size, (*color, 0))
    layer.putalpha(glow.point(lambda value: min(255, int(value * gain))))
    composed = Image.alpha_composite(base, layer)
    base.paste(composed)


def _rrect_points(box: tuple[float, float, float, float], radius: float, step: float = 6.0) -> list[tuple[float, float]]:
    left, top, right, bottom = box
    radius = min(radius, (right - left) / 2, (bottom - top) / 2)
    points: list[tuple[float, float]] = []

    def edge(x0: float, y0: float, x1: float, y1: float) -> None:
        length = math.hypot(x1 - x0, y1 - y0)
        count = max(1, int(length / step))
        for index in range(count):
            t = index / count
            points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

    def corner(cx: float, cy: float, start: float) -> None:
        count = max(4, int((math.pi / 2 * radius) / step))
        for index in range(count):
            angle = start + (math.pi / 2) * (index / count)
            points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))

    edge(left + radius, top, right - radius, top)
    corner(right - radius, top + radius, -math.pi / 2)
    edge(right, top + radius, right, bottom - radius)
    corner(right - radius, bottom - radius, 0)
    edge(right - radius, bottom, left + radius, bottom)
    corner(left + radius, bottom - radius, math.pi / 2)
    edge(left, bottom - radius, left, top + radius)
    corner(left + radius, top + radius, math.pi)
    return points


def _dashes(points: list[tuple[float, float]], on: float, off: float, width: int, size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    cycle = on + off
    travel = 0.0
    for start, end in zip(points, points[1:] + points[:1]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length < 0.1:
            continue
        steps = max(1, int(length))
        for index in range(steps):
            t = index / steps
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            if (travel % cycle) < on:
                r = width / 2
                draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
            travel += length / steps
    return mask


def _picture(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((176, 548, 458, 868), radius=46, outline=255, width=26)
    draw.ellipse((230, 610, 292, 672), outline=255, width=18)
    draw.line([(210, 820), (292, 700), (348, 778), (430, 640), (430, 820)], fill=255, width=22, joint="curve")
    return mask


def _arrow(size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    start = (530, 720)
    end = (800, 280)
    draw.line([start, end], fill=255, width=30)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    _head(draw, end, angle, 108, 255)
    _head(draw, start, angle + math.pi, 108, 255)
    return mask


def _head(draw: ImageDraw.ImageDraw, tip: tuple[float, float], angle: float, length: float, fill: int) -> None:
    spread = 0.46
    left = (tip[0] - math.cos(angle - spread) * length, tip[1] - math.sin(angle - spread) * length)
    right = (tip[0] - math.cos(angle + spread) * length, tip[1] - math.sin(angle + spread) * length)
    draw.polygon([tip, left, right], fill=fill)


def render_icon(size: int = 1024) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle((18, 18, size - 19, size - 19), radius=210, fill=(8, 14, 36, 255))
    image = Image.alpha_composite(image, tile)
    scale = size / 1024
    dashes = _dashes(_rrect_points((118, 118, 906, 906), 78, 5), on=46, off=28, width=16, size=1024)
    if size != 1024:
        dashes = dashes.resize((size, size), Image.Resampling.LANCZOS)
    # Colour the dashes along the diagonal, magenta at the top-left to coral at the bottom-right.
    stops = Image.new("RGB", (2, 2))
    stops.putpixel((0, 0), (255, 45, 196))
    stops.putpixel((1, 0), (255, 90, 150))
    stops.putpixel((0, 1), (255, 90, 150))
    stops.putpixel((1, 1), (255, 120, 86))
    gradient = stops.resize((size, size), Image.Resampling.BICUBIC)
    red, green, blue = gradient.split()
    colored = Image.merge("RGBA", (red, green, blue, dashes))
    blur = 14 * scale
    glow = colored.filter(ImageFilter.GaussianBlur(blur))
    image = Image.alpha_composite(image, glow)
    image = Image.alpha_composite(image, colored)

    picture = _picture(1024).resize((size, size), Image.Resampling.LANCZOS)
    arrow = _arrow(1024).resize((size, size), Image.Resampling.LANCZOS)
    _stamp(image, picture, (255, 70, 168), 18 * scale, 0.85)
    _stamp(image, picture, (255, 120, 190), 2 * scale, 1.0)
    _stamp(image, arrow, (255, 96, 92), 18 * scale, 0.9)
    _stamp(image, arrow, (255, 150, 120), 2 * scale, 1.0)
    return image


def render_sample() -> Image.Image:
    """A small synthetic landscape so the preview has something other than the icon."""
    width, height = 480, 320
    image = Image.new("RGB", (width, height), (20, 40, 80))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y / height
        draw.line([(0, y), (width, y)], fill=_lerp((18, 28, 72), (214, 126, 92), t * 0.85))
    draw.ellipse((330, 36, 430, 136), fill=(255, 214, 120))
    draw.polygon([(0, 250), (120, 150), (210, 230), (300, 120), (420, 220), (480, 180), (480, 320), (0, 320)], fill=(46, 62, 92))
    draw.polygon([(0, 280), (90, 210), (180, 270), (260, 190), (360, 260), (480, 210), (480, 320), (0, 320)], fill=(28, 36, 58))
    return image


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    SAMPLES.mkdir(exist_ok=True)
    icon = render_icon(1024)
    icon.save(ASSETS / "enlarge.png")
    rgb = Image.new("RGB", icon.size, (0, 0, 0))
    rgb.paste(icon.convert("RGB"), mask=icon.getchannel("A"))
    rgb.save(ASSETS / "splash.jpg", quality=90, optimize=True)
    render_sample().save(SAMPLES / "sample.png", optimize=True)
    print("wrote", ASSETS / "enlarge.png")


if __name__ == "__main__":
    main()
