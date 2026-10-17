"""Turn the shared settings into one upscale request, including a long side."""
from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from enlarge.images import reserve_output
from enlarge.ipc.scale import choose_scale, planned_size
from voxprint_engine.formats import open_rgb, save_image
from voxprint_engine.worker import UpscaleRequest


def as_int(value: object, default: int = 0) -> int:
    """Return ``value`` as an int. Bool and broken text use ``default``."""
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            text = str(value).strip()
            if not text:
                return default
            return int(text)
        except (TypeError, ValueError):
            return default
    return value


def source_size(path: Path) -> tuple[int, int]:
    """Pixel size of a still image."""
    frame = open_rgb(path)
    return frame.size


def model_scale(source: Path, scale_text: object, long_side: object) -> int:
    """Model scale, 2 or 4. A long side replaces the size chips.

    Raises:
        ValueError: The long side is larger than a 4× enlarge of this picture.
        OSError: The picture cannot be read.
    """
    side = as_int(long_side)
    if side > 0:
        width, height = source_size(source)
        return choose_scale(width, height, side)
    return 2 if str(scale_text) == "2" else 4


def fit_long_side(
    source: Path,
    dest: Path,
    scale: int,
    long_side: int,
    image_format: str,
    quality: int,
    lossless: bool,
) -> None:
    """Resize a finished file so its long side matches, when a long side was asked for."""
    if long_side < 1 or not dest.is_file():
        return
    width, height = source_size(source)
    planned = planned_size(width, height, scale, long_side)
    frame = open_rgb(dest)
    if frame.size == planned:
        return
    resized = frame.resize(planned, Image.Resampling.LANCZOS)
    save_image(resized, dest, image_format, quality, lossless)


def build_request(
    source: Path,
    destination: Path,
    *,
    scale: int,
    image_format: str,
    quality: int,
    lossless: bool,
    model_id: str,
    tile: int,
    double: bool,
    device: str,
    custom_models: str,
    app_name: str,
    job_id: str,
) -> UpscaleRequest:
    """One worker request. The destination is already reserved."""
    return UpscaleRequest(
        id=job_id,
        src=str(source),
        dst=str(destination),
        scale=scale,
        format=image_format,
        quality=quality,
        lossless=lossless,
        model_id=model_id,
        tile=tile,
        double=double,
        device=device,
        custom_models_dir=custom_models,
        app_name=app_name,
    )


@dataclass(frozen=True)
class JobSettings:
    """The values a batch reads from the shared screen at the moment Run is pressed."""

    scale_text: str
    long_side: int
    image_format: str
    quality: int
    lossless: bool
    model_id: str
    tile: int
    double: bool
    device: str
    custom_models: str
    beside: bool
    output_dir: str
    subfolders: bool


def requests_for(
    sources: list[Path],
    settings: JobSettings,
    app_name: str,
    job_ids: list[str] | None = None,
) -> list[tuple[UpscaleRequest, int]]:
    """Build one request per source. The int is the model scale, for the long-side fit.

    Raises:
        ValueError: A long side does not fit this picture.
        OSError: A picture cannot be read.
    """
    scales = [model_scale(source, settings.scale_text, settings.long_side) for source in sources]
    folder = output_folder(settings.beside, settings.output_dir)
    destinations = reserve_many(sources, scales, folder, settings.image_format, settings.double)
    built: list[tuple[UpscaleRequest, int]] = []
    for index, source in enumerate(sources):
        job_id = uuid.uuid4().hex if job_ids is None else job_ids[index]
        request = build_request(
            source,
            destinations[index],
            scale=scales[index],
            image_format=settings.image_format,
            quality=settings.quality,
            lossless=settings.lossless and settings.image_format == "jxl",
            model_id=settings.model_id,
            tile=settings.tile,
            double=settings.double,
            device=settings.device,
            custom_models=settings.custom_models,
            app_name=app_name,
            job_id=job_id,
        )
        built.append((request, scales[index]))
    return built


def output_folder(beside: object, folder_text: object) -> Path | None:
    """None saves beside the source. A path is the shared output folder."""
    if bool(beside):
        return None
    text = str(folder_text or "").strip()
    if not text:
        return None
    return Path(text)


def describe_job(source: Path, settings: JobSettings) -> list[str]:
    """The plan for one picture. No model is loaded and no file is written.

    Raises:
        ValueError: A long side does not fit this picture.
        OSError: The picture cannot be read.
    """
    scale = model_scale(source, settings.scale_text, settings.long_side)
    width, height = source_size(source)
    planned = planned_size(width, height, scale, settings.long_side)
    folder = output_folder(settings.beside, settings.output_dir)
    dest = reserve_output(source, folder, scale, settings.image_format, settings.double, set())
    return [
        f"{source} -> {dest}",
        f"{width}×{height}  scale {scale}  ->  {planned[0]}×{planned[1]}",
        f"{settings.image_format}  quality {settings.quality}  lossless {settings.lossless}",
        f"{settings.model_id}  double {settings.double}  tile {settings.tile}",
    ]


def command_line(source: Path, settings: JobSettings) -> str:
    """The headless command that matches these settings, ending in ``--dry-run``."""
    parts = ["voxprint-enlarge", "--input", str(source)]
    if settings.long_side > 0:
        parts.extend(["--long-side", str(settings.long_side)])
    else:
        parts.extend(["--scale", "2" if settings.scale_text == "2" else "4"])
    parts.extend(
        [
            "--format",
            settings.image_format,
            "--quality",
            str(settings.quality),
            "--model",
            settings.model_id,
        ]
    )
    if settings.lossless:
        parts.append("--lossless")
    if settings.double:
        parts.append("--double")
    if not settings.beside and settings.output_dir.strip():
        parts.extend(["--output-dir", settings.output_dir.strip()])
    parts.append("--dry-run")
    return " ".join(shlex.quote(part) for part in parts)


def reserve_many(
    sources: list[Path],
    scales: list[int],
    folder: Path | None,
    image_format: str,
    double: bool,
) -> list[Path]:
    """Reserve one destination per source. Scales may differ when a long side is set."""
    used: set[Path] = set()
    return [
        reserve_output(source, folder, scale, image_format, double, used)
        for source, scale in zip(sources, scales, strict=True)
    ]
