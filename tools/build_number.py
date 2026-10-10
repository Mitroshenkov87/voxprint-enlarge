"""Build number for a release. ``GITHUB_RUN_NUMBER`` plus ``offset`` in ``BUILD.json``.

A local run without ``GITHUB_RUN_NUMBER`` is build 0. ``VOXPRINT_BUILD`` overrides both.
``codename`` is one Biblical Hebrew word in Latin letters, edited in ``BUILD.json``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def info() -> dict[str, object]:
    data = json.loads((ROOT / "BUILD.json").read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_number(env: dict[str, str] | None = None) -> int:
    source = os.environ if env is None else env
    override = str(source.get("VOXPRINT_BUILD", "")).strip()
    if override.isdigit():
        return int(override)
    run = str(source.get("GITHUB_RUN_NUMBER", "")).strip()
    if not run.isdigit():
        return 0
    raw_offset = info().get("offset", 0)
    try:
        offset = int(raw_offset)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        offset = 0
    return int(run) + offset


def codename() -> str:
    return str(info().get("codename") or "").strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    number = build_number()
    if "--codename" in args:
        print(codename())
        return 0
    print(number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
