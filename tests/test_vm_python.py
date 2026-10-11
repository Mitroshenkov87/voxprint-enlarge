"""The VM smoke script must not call the Microsoft Store python3 alias first."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vm_test.sh"
PS1 = ROOT / "installer" / "windows" / "start-menu.ps1"


def _resolve(env: dict[str, str]) -> str:
    merged = os.environ.copy()
    merged.update(env)
    merged.pop("VOXPRINT_HOME", None)
    merged.pop("LOCALAPPDATA", None)
    merged.pop("XDG_DATA_HOME", None)
    for key, value in env.items():
        if value == "":
            merged.pop(key, None)
        else:
            merged[key] = value
    completed = subprocess.run(
        ["bash", "-c", 'source "$1"; resolve_vm_python', "bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=merged,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_shared_runtime_wins_over_py_and_python3(tmp_path: Path) -> None:
    home = tmp_path / "home"
    runtime = home / "shared" / "python" / "bin"
    runtime.mkdir(parents=True)
    chosen = runtime / "python3"
    chosen.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    chosen.chmod(0o755)
    found = _resolve({"VOXPRINT_HOME": str(home), "PATH": "/usr/bin"})
    assert found == str(chosen)


def test_py_launcher_is_before_python3(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    py = bindir / "py"
    py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)
    python3 = bindir / "python3"
    python3.write_text("#!/bin/sh\necho store >&2\nexit 1\n", encoding="utf-8")
    python3.chmod(0o755)
    found = _resolve({"PATH": f"{bindir}:/usr/bin", "HOME": str(tmp_path / "empty")})
    assert found == "py -3"


def test_python3_is_the_last_resort(tmp_path: Path) -> None:
    found = _resolve({"PATH": "/usr/bin", "HOME": str(tmp_path / "empty")})
    assert found == "python3"


def test_start_menu_script_uses_the_same_order() -> None:
    text = PS1.read_text(encoding="utf-8")
    shared = text.index("shared\\python\\pythonw.exe")
    launcher = text.index("py.exe")
    python3 = text.index("python3.exe")
    assert shared < launcher < python3
    assert "WindowsApps" in text
