"""Build number for a release, from ``voxprint_common.buildinfo``.

``GITHUB_RUN_NUMBER`` plus ``offset`` in ``BUILD.json``. ``VOXPRINT_BUILD``
overrides both. Without a run number, the recorded ``build`` is used. A
missing recorded build stays null. It is not reported as 0.
``codename`` is one Biblical Hebrew word in Latin letters, edited in ``BUILD.json``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from voxprint_common.buildinfo import load

ROOT = Path(__file__).resolve().parent.parent


def build_number(env: dict[str, str] | None = None) -> int | None:
    """The release build, or ``None`` when this tree has no recorded build."""
    source = os.environ if env is None else env
    return load(ROOT / "BUILD.json", app="enlarge", env=source).build


def codename() -> str:
    """Codename from ``BUILD.json``."""
    return load(ROOT / "BUILD.json", app="enlarge").codename


def main(argv: list[str] | None = None) -> int:
    """Print the build number, or ``dev`` when it is null, or the codename."""
    args = sys.argv[1:] if argv is None else argv
    if "--codename" in args:
        print(codename())
        return 0
    number = build_number()
    print("dev" if number is None else number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
