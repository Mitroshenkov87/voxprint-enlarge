"""Open VOXPrint MDL. Enlarge does not download model files itself."""
from __future__ import annotations

import subprocess
import sys

from voxprint_engine.process_win import no_window_flags


def manage_models_command() -> list[str]:
    """The Manage models button. MDL's window lists Enlarge's models."""
    return ["voxprint-mdl", "--gui", "--app", "enlarge"]


def start_manage_models() -> subprocess.Popen[bytes]:
    """Launch ``voxprint-mdl --gui --app enlarge``.

    When that command is not on PATH, the current interpreter runs the module.
    """
    command = manage_models_command()
    try:
        return subprocess.Popen(command, creationflags=no_window_flags())
    except OSError:
        fallback = [sys.executable, "-m", "voxprint_common.mdl", "--gui", "--app", "enlarge"]
        return subprocess.Popen(fallback, creationflags=no_window_flags())
