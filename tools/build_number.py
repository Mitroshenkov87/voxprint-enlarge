"""Build number for a release.

The release workflow runs this module before it installs dependencies, so it
does not import ``voxprint_common``. The rule matches voxprint-common 0.2.4:
``VOXPRINT_BUILD`` wins, otherwise ``GITHUB_RUN_NUMBER`` plus ``offset`` in
``BUILD.json``, otherwise the recorded ``build`` (from ``BUILD.json`` or a
sibling ``build_info.json``). A missing recorded build stays null. It is not
reported as 0. ``codename`` is one Biblical Hebrew word in Latin letters,
edited in ``BUILD.json``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_number(env: dict[str, str] | None = None) -> int | None:
    """The release build, or ``None`` when this tree has no recorded build."""
    source = os.environ if env is None else env
    data = _document()
    override = str(source.get("VOXPRINT_BUILD", "")).strip()
    if override.isdigit():
        return int(override)
    run = str(source.get("GITHUB_RUN_NUMBER", "")).strip()
    if run.isdigit():
        offset = _as_int(data.get("offset")) or 0
        return int(run) + offset
    return _recorded(data)


def codename() -> str:
    """Codename from ``BUILD.json``."""
    return _codename(_document().get("codename"))


def main(argv: list[str] | None = None) -> int:
    """Print the build number, or ``dev`` when it is null, or the codename."""
    args = sys.argv[1:] if argv is None else argv
    if "--codename" in args:
        print(codename())
        return 0
    number = build_number()
    print("dev" if number is None else number)
    return 0


def _document() -> dict[str, object]:
    """``BUILD.json``, with a sibling ``build_info.json`` supplying the recorded build."""
    catalog = _read(ROOT / "BUILD.json")
    frozen_path = ROOT / "build_info.json"
    if not frozen_path.is_file():
        return catalog
    frozen = _read(frozen_path)
    offset = catalog.get("offset") if "offset" in catalog else None
    catalog_name = catalog.get("codename")
    for key, value in frozen.items():
        if value is None:
            continue
        catalog[key] = value
    if offset is not None:
        catalog["offset"] = offset
    frozen_name = frozen.get("codename")
    frozen_blank = not isinstance(frozen_name, str) or not frozen_name.strip()
    if frozen_blank and isinstance(catalog_name, str) and catalog_name.strip():
        catalog["codename"] = catalog_name
    return catalog


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _recorded(data: dict[str, object]) -> int | None:
    recorded = _as_int(data.get("build")) if "build" in data else None
    if recorded is None and "number" in data:
        return _as_int(data.get("number"))
    return recorded


def _codename(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


if __name__ == "__main__":
    raise SystemExit(main())
