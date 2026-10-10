"""Output formats: AVIF, JPEG XL, and WebP. PNG is an input. HEIC is optional."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from enlarge import settings
from voxprint_engine.errors import ImageJobError
from voxprint_engine.formats import (
    DEFAULT_FORMAT,
    DEFAULT_QUALITY,
    LEGACY_IDS,
    choices,
    heic_backend,
    normalize_format,
    open_rgb,
    require_output_format,
    reset_heic_cache,
    save_image,
)


def _frame() -> Image.Image:
    image = Image.new("RGB", (32, 24))
    image.putdata([(x * 7 % 256, y * 11 % 256, (x + y) * 5 % 256) for y in range(24) for x in range(32)])
    return image


def _max_delta(left: Image.Image, right: Image.Image) -> int:
    delta = np.abs(np.asarray(left).astype(np.int16) - np.asarray(right).astype(np.int16))
    return int(delta.max())


def test_main_set_and_legacy_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voxprint_engine.formats.detect_backend", lambda: None)
    reset_heic_cache()
    rows = choices()
    ids = [key for _label, key in rows]
    assert DEFAULT_FORMAT == "avif"
    assert DEFAULT_FORMAT not in LEGACY_IDS
    assert ids[0] == "avif"
    assert ids == ["avif", "jxl", "webp", "jpeg"]
    assert rows[3] == ("JPEG (legacy)", "jpeg")
    assert "png" not in ids
    assert "heic" not in ids
    assert normalize_format("png") == "avif"
    assert normalize_format("jpg") == "jpeg"
    assert normalize_format("nope") == "avif"
    assert DEFAULT_QUALITY == 60
    reset_heic_cache()


def test_heic_appears_only_with_a_system_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voxprint_engine.formats.detect_backend", lambda: None)
    reset_heic_cache()
    with pytest.raises(ImageJobError, match="system codec"):
        require_output_format("heic", lossless=False)
    monkeypatch.setattr("voxprint_engine.formats.detect_backend", lambda: "libheif")
    reset_heic_cache()
    assert heic_backend() == "libheif"
    assert choices()[-1] == ("HEIC (legacy)", "heic")
    assert require_output_format("heic", lossless=False) == "heic"
    with pytest.raises(ImageJobError, match="JPEG XL only"):
        require_output_format("heic", lossless=True)
    reset_heic_cache()


def test_png_is_rejected_and_lossless_is_jpeg_xl_only() -> None:
    with pytest.raises(ImageJobError, match="PNG is an input"):
        require_output_format("png", lossless=False)
    with pytest.raises(ImageJobError, match="JPEG XL only"):
        require_output_format("avif", lossless=True)
    with pytest.raises(ImageJobError, match="JPEG XL only"):
        require_output_format("webp", lossless=True)
    assert require_output_format("jxl", lossless=True) == "jxl"


def test_old_png_setting_falls_back_to_avif() -> None:
    target = settings.path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"format": "png", "quality": 95, "lossless": true}\n', encoding="utf-8")
    loaded = settings.load()
    assert loaded["format"] == "avif"
    assert loaded["quality"] == 95
    assert loaded["lossless"] is False


def _mean_delta(left: Image.Image, right: Image.Image) -> float:
    delta = np.abs(np.asarray(left).astype(np.int16) - np.asarray(right).astype(np.int16))
    return float(delta.mean())


def test_round_trip(tmp_path: Path) -> None:
    flat = Image.new("RGB", (32, 24), (10, 20, 30))
    varied = _frame()
    magics = {
        "avif": b"ftypavif",
        "webp": b"WEBP",
        "jpeg": b"\xff\xd8",
        "jxl": b"\xff\x0a",
    }
    for image_format, suffix in (("avif", ".avif"), ("webp", ".webp"), ("jpeg", ".jpg"), ("jxl", ".jxl")):
        plain = tmp_path / f"flat{suffix}"
        save_image(flat, plain, image_format, 60, lossless=False)
        assert _max_delta(flat, open_rgb(plain)) <= 20
        dest = tmp_path / f"varied{suffix}"
        save_image(varied, dest, image_format, 60, lossless=False)
        back = open_rgb(dest)
        assert back.size == varied.size
        assert _mean_delta(varied, back) <= 15
        blob = dest.read_bytes()
        assert magics[image_format] in blob[:32]
    exact = tmp_path / "exact.jxl"
    save_image(varied, exact, "jxl", 60, lossless=True)
    assert np.array_equal(np.asarray(open_rgb(exact)), np.asarray(varied))
    source = tmp_path / "in.png"
    varied.save(source)
    assert np.array_equal(np.asarray(open_rgb(source)), np.asarray(varied))


def test_no_bundled_heic_or_avif_plugin() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "pillow-heif" not in requirements
    assert "pillow_heif" not in requirements
    assert "pillow-jxl" not in requirements
    assert "pillow-avif" not in requirements
    assert "pillow_avif" not in requirements
    assert "Pillow>=11.3" in requirements
    for path in Path("voxprint_engine").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import pillow_heif" not in text
        assert "from pillow_heif" not in text
        assert "import pillow_jxl" not in text
        assert "pillow_avif" not in text
        assert "pillow-avif" not in text
