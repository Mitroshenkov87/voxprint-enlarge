"""The VM smoke script must not call the Microsoft Store python3 alias first."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vm_test.sh"
PS1 = ROOT / "installer" / "windows" / "start-menu.ps1"


def _bash_path(path: Path) -> str:
    """Path Git bash can open. Drive letters stay colon-free (`/c/Users`)."""
    text = path.as_posix()
    if os.name == "nt" and len(text) >= 3 and text[1] == ":":
        return f"/{text[0].lower()}{text[2:]}"
    return text


def _resolve(env: dict[str, str]) -> str:
    merged = os.environ.copy()
    for key in ("VOXPRINT_HOME", "LOCALAPPDATA", "XDG_DATA_HOME", "BASH_ENV", "SHELLOPTS", "BASHOPTS"):
        merged.pop(key, None)
    for key, value in env.items():
        if value == "":
            merged.pop(key, None)
        else:
            merged[key] = value
    # Windows keeps `Path` and `PATH` as one variable. Set both spellings.
    if "PATH" in env and os.name == "nt":
        merged["Path"] = env["PATH"]
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", 'source "$1"; resolve_vm_python', "vosource", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=merged,
    )
    assert completed.returncode == 0, (completed.returncode, completed.stdout, completed.stderr)
    return completed.stdout.strip()


def test_shared_runtime_wins_over_py_and_python3(tmp_path: Path) -> None:
    home = tmp_path / "home"
    runtime = home / "shared" / "python" / "bin"
    runtime.mkdir(parents=True)
    chosen = runtime / "python3"
    chosen.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    chosen.chmod(0o755)
    windows_home = str(home).replace("/", "\\")
    found = _resolve({"VOXPRINT_HOME": windows_home, "PATH": "/usr/bin", "HOME": _bash_path(tmp_path / "empty-home")})
    assert found.replace("\\", "/") == str(chosen).replace("\\", "/")


def test_py_launcher_is_before_python3(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    py = bindir / "py"
    py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)
    python3 = bindir / "python3"
    python3.write_text("#!/bin/sh\necho store >&2\nexit 1\n", encoding="utf-8")
    python3.chmod(0o755)
    bash_bin = _bash_path(bindir)
    subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", 'chmod +x "$1" "$2"', "chmod", f"{bash_bin}/py", f"{bash_bin}/python3"],
        check=True,
    )
    found = _resolve({"PATH": f"{bash_bin}:/usr/bin", "HOME": _bash_path(tmp_path / "empty")})
    assert found == "py -3"


def test_python3_is_the_last_resort(tmp_path: Path) -> None:
    found = _resolve({"PATH": "/usr/bin", "HOME": _bash_path(tmp_path / "empty")})
    assert found == "python3"


def test_start_menu_script_uses_the_same_order() -> None:
    text = PS1.read_text(encoding="utf-8")
    shared = text.index("shared\\python\\pythonw.exe")
    launcher = text.index("py.exe")
    python3 = text.index("python3.exe")
    assert shared < launcher < python3
    assert "WindowsApps" in text
