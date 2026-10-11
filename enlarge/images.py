"""Image paths only. Video files are refused with a short explanation."""
from __future__ import annotations

from pathlib import Path

from voxprint_engine.formats import output_extension
from voxprint_engine.worker.lanczos import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, output_scale


def is_image(path: Path) -> bool:
    """True when the path has an image suffix this app can open."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect(paths: list[Path], *, recursive: bool) -> tuple[list[Path], list[str]]:
    """Expand files and folders into images. Warnings explain skips; they are not failures."""
    found: list[Path] = []
    warnings: list[str] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(path.glob(pattern)):
                if not child.is_file():
                    continue
                suffix = child.suffix.lower()
                if suffix in VIDEO_EXTENSIONS:
                    warnings.append(f"Skipped {child.name}: Enlarge is for images. Video belongs to VOXPrint AI Clarity.")
                    continue
                if suffix in IMAGE_EXTENSIONS:
                    add(child)
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            warnings.append(f"Skipped {path.name}: Enlarge is for images. Video belongs to VOXPrint AI Clarity.")
            continue
        if path.is_file() and suffix in IMAGE_EXTENSIONS:
            add(path)
        elif path.exists():
            warnings.append(f"Skipped {path.name}: not a still image.")
        else:
            warnings.append(f"Skipped {path.name}: file not found.")
    return found, warnings


def plan_outputs(
    sources: list[Path],
    folder: Path | None,
    scale: int,
    image_format: str,
    double: bool,
) -> list[Path]:
    """One output path per source. Collisions get a numeric suffix before the extension."""
    factor = output_scale(scale, double)
    ext = output_extension(image_format)
    used: set[Path] = set()
    planned: list[Path] = []
    for src in sources:
        directory = folder if folder is not None else src.parent
        candidate = directory / f"{src.stem}_x{factor}{ext}"
        number = 2
        while candidate in used or candidate.exists():
            candidate = directory / f"{src.stem}_x{factor}-{number}{ext}"
            number += 1
        used.add(candidate)
        planned.append(candidate)
    return planned
