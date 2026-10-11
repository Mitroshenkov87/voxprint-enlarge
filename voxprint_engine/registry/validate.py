"""Licence gate for model manifests.

Release builds accept only the families named in the Enlarge decision:
CC-BY, Apache-2.0, MIT, BSD-2/3, and CC0. Non-commercial and missing licences
are rejected in every build, so a custom-models folder cannot quietly add them.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from voxprint_engine.registry.schema import ModelManifest, checked_in_channels

ROLES = ("upscale", "restore", "face", "generative")
SCALES = (1, 2, 4)
EXECUTIONS = ("auto", "cuda", "tensorrt")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,80}$")
_ARCH = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,40}$")
_SHA = re.compile(r"^[a-f0-9]{64}$")
_NC = re.compile(r"(^|[^a-z0-9])nc([^a-z0-9]|$)")

#: SPDX ids allowed when ``release`` is true. Aliases are folded before this check.
ALLOWED_RELEASE = frozenset({
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-2.0",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc0-1.0",
    "isc",
})

_UNLICENSED = frozenset({
    "",
    "unlicensed",
    "none",
    "unknown",
    "proprietary",
    "all-rights-reserved",
    "no-license",
    "missing",
    "tbd",
})

_ALIASES = {
    "apache-2": "apache-2.0",
    "apache2": "apache-2.0",
    "apache2.0": "apache-2.0",
    "bsd-2": "bsd-2-clause",
    "bsd-3": "bsd-3-clause",
    "bsd": "bsd-3-clause",
    "cc-by": "cc-by-4.0",
    "cc-by-0.4": "cc-by-4.0",  # typo used on some model cards; Hugging Face stores cc-by-4.0
}


class ManifestError(ValueError):
    """One or more manifest fields failed the gate. ``errors`` is the full list."""

    def __init__(self, errors: list[str]) -> None:
        """Store the list of reasons the manifest was rejected."""
        self.errors = errors
        super().__init__("; ".join(errors))


def normalize_licence(value: object) -> str:
    """Fold case, separators, and a few aliases. Empty stays empty."""
    if not isinstance(value, str):
        return ""
    text = value.strip().lower().replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return _ALIASES.get(text, text)


def is_unlicensed(licence: str) -> bool:
    """True when the licence string is empty or non-commercial."""
    return normalize_licence(licence) in _UNLICENSED


def is_noncommercial(licence: str) -> bool:
    """True for CC-BY-NC*, 'non-commercial', and a bare ``nc`` token."""
    text = normalize_licence(licence)
    if not text:
        return False
    if "non-commercial" in text or "noncommercial" in text:
        return True
    return _NC.search(text) is not None


def canonical_licence(licence: str) -> str:
    """Display form stored on the manifest (SPDX-ish, original casing for known ids)."""
    text = normalize_licence(licence)
    pretty = {
        "mit": "MIT",
        "apache-2.0": "Apache-2.0",
        "bsd-2-clause": "BSD-2-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "cc-by-2.0": "CC-BY-2.0",
        "cc-by-3.0": "CC-BY-3.0",
        "cc-by-4.0": "CC-BY-4.0",
        "cc0-1.0": "CC0-1.0",
        "isc": "ISC",
    }
    return pretty.get(text, text)


def _str(data: Mapping[str, Any], key: str, errors: list[str], *, max_len: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} is required")
        return ""
    text = value.strip()
    if len(text) > max_len:
        errors.append(f"{key} is longer than {max_len} characters")
    return text


def _int(data: Mapping[str, Any], key: str, errors: list[str]) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{key} must be an integer")
        return 0
    return value


def validate_manifest(data: Mapping[str, Any], *, release: bool) -> ModelManifest:
    """Parse ``data`` into a manifest or raise :class:`ManifestError`.

    ``release`` True uses the allow-list. False still rejects non-commercial
    and unlicensed entries, and still checks the shape of every field.
    """
    if not isinstance(data, Mapping):
        raise ManifestError(["manifest must be an object"])
    errors: list[str] = []
    model_id = _str(data, "id", errors, max_len=81)
    if model_id and not _ID.match(model_id):
        errors.append("id must be lowercase words separated by hyphens")
    name = _str(data, "name", errors, max_len=120)
    arch = _str(data, "arch", errors, max_len=41).lower()
    if arch and not _ARCH.match(arch):
        errors.append("arch has unexpected characters")
    scale = _int(data, "scale", errors)
    if "scale must be an integer" not in " ".join(errors) and scale not in SCALES:
        errors.append("scale must be 1, 2, or 4")
    role = _str(data, "role", errors, max_len=20).lower()
    if role and role not in ROLES:
        errors.append("role must be upscale, restore, face, or generative")
    raw_licence = data.get("licence", "")
    folded = normalize_licence(raw_licence)
    if is_unlicensed(folded):
        errors.append("licence is missing or unlicensed")
    elif is_noncommercial(folded):
        errors.append(f"non-commercial licence {raw_licence!r} is not accepted")
    elif release and folded not in ALLOWED_RELEASE:
        errors.append(f"licence {raw_licence!r} is not allowed in a release build")
    author = _str(data, "author", errors, max_len=160)
    attribution = _str(data, "attribution", errors, max_len=500)
    source = _str(data, "source_url", errors, max_len=300)
    if source and not (source.startswith("https://") or source.startswith("http://")):
        errors.append("source_url must be an http(s) URL")
    sha = data.get("sha256", "")
    if sha is None:
        sha = ""
    if not isinstance(sha, str):
        errors.append("sha256 must be a string")
        sha = ""
    sha = sha.strip().lower()
    if sha and not _SHA.match(sha):
        errors.append("sha256 must be 64 lowercase hex characters, or empty until the file is pinned")
    vram_mb = _int(data, "vram_mb", errors)
    if vram_mb < 0:
        errors.append("vram_mb must be >= 0")
    tile = _int(data, "tile", errors)
    if tile < 0:
        errors.append("tile must be >= 0")
    filename = _str(data, "filename", errors, max_len=160)
    summary = _str(data, "summary", errors, max_len=400)
    download = data.get("download_url", "")
    if download is None:
        download = ""
    if not isinstance(download, str):
        errors.append("download_url must be a string")
        download = ""
    download = download.strip()
    if download and not download.startswith("https://"):
        errors.append("download_url must be an https URL")
    if len(download) > 300:
        errors.append("download_url is longer than 300 characters")
    if download and not sha:
        errors.append("sha256 is required when download_url is set")
    onnx_url = data.get("onnx_url", "")
    if onnx_url is None:
        onnx_url = ""
    if not isinstance(onnx_url, str):
        errors.append("onnx_url must be a string")
        onnx_url = ""
    onnx_url = onnx_url.strip()
    if onnx_url and not onnx_url.startswith("https://"):
        errors.append("onnx_url must be an https URL")
    if len(onnx_url) > 300:
        errors.append("onnx_url is longer than 300 characters")
    onnx_sha = data.get("onnx_sha256", "")
    if onnx_sha is None:
        onnx_sha = ""
    if not isinstance(onnx_sha, str):
        errors.append("onnx_sha256 must be a string")
        onnx_sha = ""
    onnx_sha = onnx_sha.strip().lower()
    if onnx_sha and not _SHA.match(onnx_sha):
        errors.append("onnx_sha256 must be 64 lowercase hex characters")
    if onnx_url and not onnx_sha:
        errors.append("onnx_sha256 is required when onnx_url is set")
    if onnx_sha and not onnx_url:
        errors.append("onnx_url is required when onnx_sha256 is set")
    execution = data.get("execution", "auto")
    if execution is None or execution == "":
        execution = "auto"
    if not isinstance(execution, str):
        errors.append("execution must be auto, cuda, or tensorrt")
        execution = "auto"
    else:
        execution = execution.strip().lower()
        if execution not in EXECUTIONS:
            errors.append("execution must be auto, cuda, or tensorrt")
    if "in_channels" not in data or data.get("in_channels") is None:
        in_channels = 3
    else:
        try:
            in_channels = checked_in_channels(data.get("in_channels"))
        except ValueError as exc:
            errors.append(str(exc))
            in_channels = 3
    selectable = data.get("selectable", True)
    if not isinstance(selectable, bool):
        errors.append("selectable must be true or false")
        selectable = False
    if errors:
        raise ManifestError(errors)
    return ModelManifest(
        id=model_id,
        name=name,
        arch=arch,
        scale=scale,
        role=role,
        licence=canonical_licence(folded),
        author=author,
        attribution=attribution,
        source_url=source,
        sha256=sha,
        vram_mb=vram_mb,
        tile=tile,
        filename=filename,
        summary=summary,
        selectable=selectable,
        download_url=download,
        onnx_url=onnx_url,
        onnx_sha256=onnx_sha,
        execution=execution,
        in_channels=in_channels,
    )
