"""Registry shape and the release licence gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxprint_engine.registry import DEFAULT_MODEL_ID, load_builtin, validate_manifest
from voxprint_engine.registry.catalog import combined_catalog
from voxprint_engine.registry.validate import ManifestError, is_noncommercial, normalize_licence

ALLOWED = ("CC-BY-4.0", "Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "CC0-1.0")
BLOCKED = ("CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "UNLICENSED", "", "proprietary", "unknown")


def _manifest(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "4x-example-model",
        "name": "Example",
        "arch": "rrdb",
        "scale": 4,
        "role": "upscale",
        "licence": "MIT",
        "author": "Someone",
        "attribution": "Example by Someone, MIT.",
        "source_url": "https://example.com/model",
        "sha256": "",
        "vram_mb": 1000,
        "tile": 256,
        "filename": "example.pth",
        "summary": "A test manifest.",
        "selectable": True,
    }
    data.update(overrides)
    return data


def test_builtin_catalog_is_release_safe() -> None:
    catalog = load_builtin(release=True)
    assert catalog.get(DEFAULT_MODEL_ID) is not None
    assert catalog.get(DEFAULT_MODEL_ID).selectable  # type: ignore[union-attr]
    credits = catalog.credits()
    assert credits
    assert all(model.licence == "CC-BY-4.0" for model in credits)
    assert any(model.author == "Philip Hofmann" for model in credits)
    assert all(model.selectable for model in catalog.selectable())
    assert any(not model.selectable for model in catalog.models)
    ids = [model.id for model in catalog.models]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("licence", ALLOWED)
def test_release_accepts_safe_licences(licence: str) -> None:
    model = validate_manifest(_manifest(licence=licence), release=True)
    assert model.licence


@pytest.mark.parametrize("licence", BLOCKED)
def test_noncommercial_and_unlicensed_are_rejected(licence: str) -> None:
    with pytest.raises(ManifestError):
        validate_manifest(_manifest(licence=licence), release=True)
    with pytest.raises(ManifestError):
        validate_manifest(_manifest(licence=licence), release=False)


def test_release_rejects_licences_outside_the_allow_list() -> None:
    with pytest.raises(ManifestError, match="not allowed in a release"):
        validate_manifest(_manifest(licence="GPL-3.0"), release=True)
    model = validate_manifest(_manifest(licence="GPL-3.0"), release=False)
    assert model.licence == "gpl-3.0"


def test_bad_sha_and_scale_are_rejected() -> None:
    with pytest.raises(ManifestError, match="sha256"):
        validate_manifest(_manifest(sha256="abcd"), release=True)
    with pytest.raises(ManifestError, match="scale"):
        validate_manifest(_manifest(scale=3), release=True)
    ok = validate_manifest(_manifest(sha256="a" * 64), release=True)
    assert ok.sha256 == "a" * 64


def test_nc_alias_is_noncommercial() -> None:
    assert is_noncommercial("CC BY-NC 4.0")
    assert normalize_licence("CC-BY-0.4") == "cc-by-4.0"
    assert not is_noncommercial("Apache-2.0")


def test_custom_folder_skips_a_blocked_manifest(tmp_path: Path) -> None:
    bad = _manifest(id="4x-blocked-model", licence="CC-BY-NC-SA-4.0")
    good = _manifest(id="4x-local-ok", licence="Apache-2.0", name="Local")
    (tmp_path / "extra.json").write_text(json.dumps({"models": [bad, good]}), encoding="utf-8")
    catalog = combined_catalog(tmp_path, release=True)
    assert catalog.get("4x-local-ok") is not None
    assert catalog.get("4x-blocked-model") is None
    assert catalog.warnings
    assert DEFAULT_MODEL_ID in {model.id for model in catalog.models}


def test_duplicate_custom_id_does_not_replace_builtin(tmp_path: Path) -> None:
    stolen = _manifest(id=DEFAULT_MODEL_ID, licence="MIT", author="Not the author")
    (tmp_path / "override.json").write_text(json.dumps(stolen), encoding="utf-8")
    catalog = combined_catalog(tmp_path, release=True)
    model = catalog.get(DEFAULT_MODEL_ID)
    assert model is not None
    assert model.author == "Philip Hofmann"
    assert any("duplicate" in warning for warning in catalog.warnings)
