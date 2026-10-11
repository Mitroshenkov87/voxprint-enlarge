"""Start a worker without opening a console window on Windows."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def gui_python(executable: str | None = None) -> str:
    """``pythonw`` beside ``python.exe`` on Windows, otherwise the given interpreter.

    A console ``python.exe`` child opens its own window when the parent is the
    GUI. ``pythonw.exe`` does not. The child still inherits piped stdout.
    """
    chosen = executable or sys.executable
    if os.name != "nt":
        return chosen
    path = Path(chosen)
    if path.name.lower() != "python.exe":
        return chosen
    sibling = path.with_name("pythonw.exe")
    if sibling.is_file():
        return str(sibling)
    return chosen


def no_window_flags() -> int:
    """``CREATE_NO_WINDOW`` on Windows, otherwise 0."""
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
