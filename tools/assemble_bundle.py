"""Stage the source tree that the Windows installer and the Linux tarball ship.

Weights are not copied. ``--with-python`` downloads a relocatable CPython and
installs ``requirements.txt`` into it. The installer then runs ``--install-suite``,
which copies that runtime into the shared suite folder and installs the CUDA
wheels and the default model on the machine.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tools.build_number import build_number, codename

ROOT = Path(__file__).resolve().parent.parent


def release_version() -> str:
    """Product version stamped into ``build_info.json``. It is the ``version`` field of ``BUILD.json``."""
    payload = json.loads((ROOT / "BUILD.json").read_text(encoding="utf-8"))
    version = str(payload.get("version") or "").strip() if isinstance(payload, dict) else ""
    if not version:
        raise SystemExit("BUILD.json needs a version")
    return version
DIRS = ("enlarge", "voxprint_engine", "assets", "samples", "docs")
FILES = ("main.py", "LICENSE", "NOTICE", "README.md", "requirements.txt", "requirements-gpu.txt", "BUILD.json")
SKIP_SUFFIXES = {".pth", ".onnx", ".pyc"}


def _ignore(folder: str, names: list[str]) -> set[str]:
    del folder
    skipped = {"__pycache__", ".pytest_cache"}
    for name in names:
        if Path(name).suffix.lower() in SKIP_SUFFIXES:
            skipped.add(name)
    return skipped


def stage_app(dest: Path) -> None:
    """Copy the app. Model weights and ONNX files are left behind."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for folder in DIRS:
        shutil.copytree(ROOT / folder, dest / folder, ignore=_ignore)
    for name in FILES:
        shutil.copy2(ROOT / name, dest / name)
    installer = dest / "installer" / "linux"
    installer.mkdir(parents=True)
    shutil.copy2(ROOT / "installer" / "linux" / "voxprint-enlarge.desktop", installer / "voxprint-enlarge.desktop")
    icon = ROOT / "installer" / "linux" / "voxprint-enlarge-256.png"
    if icon.is_file():
        shutil.copy2(icon, installer / "voxprint-enlarge-256.png")
    (dest / "install.sh").write_text(_install_sh(), encoding="utf-8")
    (dest / "uninstall.sh").write_text(_uninstall_sh(), encoding="utf-8")
    launcher = dest / "voxprint-enlarge"
    launcher.write_text(_linux_launcher(), encoding="utf-8")
    launcher.chmod(0o755)
    (dest / "install.sh").chmod(0o755)
    (dest / "uninstall.sh").chmod(0o755)


def stamp_build_info(dest: Path, env: dict[str, str] | None = None) -> None:
    source = os.environ if env is None else env
    payload = {
        "version": release_version(),
        "build": build_number(source),
        "codename": codename(),
        "commit": str(source.get("GITHUB_SHA", ""))[:12],
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (dest / "build_info.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _linux_launcher() -> str:
    return """#!/bin/sh
ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
SHARED="${XDG_DATA_HOME:-$HOME/.local/share}/voxprint/shared/python/bin/python3"
BUNDLED="$ROOT/python/bin/python3"
# A shared interpreter left by an older suite install can exist and still
# lack voxprint_common or onnxruntime. Fall back to the bundle.
if [ -x "$SHARED" ] && "$SHARED" -c "import voxprint_common, voxprint_engine, onnxruntime" >/dev/null 2>&1; then
  exec "$SHARED" "$ROOT/main.py" "$@"
fi
exec "$BUNDLED" "$ROOT/main.py" "$@"
"""


def _install_sh() -> str:
    return """#!/bin/sh
set -eu

target_home() {
  if [ -n "${SUDO_USER:-}" ]; then
    home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
  else
    home=$HOME
  fi
  if [ -z "$home" ] || [ "$home" = "/" ]; then
    echo "install: could not resolve the user home" >&2
    exit 1
  fi
  printf '%s\\n' "$home"
}

run_as_user() {
  if [ -n "${SUDO_USER:-}" ]; then
    sudo -u "$SUDO_USER" -H -- "$@"
  else
    "$@"
  fi
}

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -- "$0" "$@"
fi
ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
DEST=/opt/voxprint-ai-suite/enlarge
mkdir -p "$DEST"
cp -a "$ROOT"/. "$DEST"/
chmod -R a+rX "$DEST"
chmod 755 "$DEST/voxprint-enlarge" "$DEST/install.sh" "$DEST/uninstall.sh"
HOME_DIR=$(target_home)
DATA="$HOME_DIR/.local/share"
OLD="$DATA/voxprint/enlarge"
# The old tree is the app only. shared/, models, and logs stay.
case "$OLD" in
  */voxprint/enlarge)
    rm -rf "$OLD"
    ;;
esac
rm -f "$DATA/applications/voxprint-enlarge.desktop"
rm -f "$HOME_DIR/.local/bin/voxprint-enlarge"
run_as_user "$DEST/python/bin/python3" "$DEST/main.py" --install-suite
run_as_user "$DEST/python/bin/python3" "$DEST/main.py" --register --exe "$DEST/voxprint-enlarge"
mkdir -p /usr/share/applications
cat > /usr/share/applications/voxprint-enlarge.desktop <<EOF
[Desktop Entry]
Type=Application
Name=VOXPrint AI Enlarge
Comment=Offline image upscaling
Exec=$DEST/voxprint-enlarge
Icon=$DEST/assets/enlarge.png
Terminal=false
Categories=Graphics;
StartupWMClass=VOXPrint AI Enlarge
EOF
ln -sfn "$DEST/voxprint-enlarge" /usr/local/bin/voxprint-enlarge
echo "$DEST"
"""


def _uninstall_sh() -> str:
    return """#!/bin/sh
set -eu

target_home() {
  if [ -n "${SUDO_USER:-}" ]; then
    home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
  else
    home=$HOME
  fi
  if [ -z "$home" ] || [ "$home" = "/" ]; then
    echo "uninstall: could not resolve the user home" >&2
    exit 1
  fi
  printf '%s\\n' "$home"
}

run_as_user() {
  if [ -n "${SUDO_USER:-}" ]; then
    sudo -u "$SUDO_USER" -H -- "$@"
  else
    "$@"
  fi
}

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -- "$0" "$@"
fi
ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
run_as_user "$ROOT/python/bin/python3" "$ROOT/main.py" --uninstall-suite
run_as_user "$ROOT/python/bin/python3" "$ROOT/main.py" --unregister
HOME_DIR=$(target_home)
DATA="$HOME_DIR/.local/share"
OLD="$DATA/voxprint/enlarge"
case "$OLD" in
  */voxprint/enlarge)
    rm -rf "$OLD"
    ;;
esac
rm -f "$DATA/applications/voxprint-enlarge.desktop"
rm -f "$HOME_DIR/.local/bin/voxprint-enlarge"
rm -f /usr/share/applications/voxprint-enlarge.desktop
rm -f /usr/local/bin/voxprint-enlarge
if [ -d /opt/voxprint-ai-suite/enlarge ]; then
  rm -rf /opt/voxprint-ai-suite/enlarge
fi
"""


def python_asset_name(names: list[str], platform: str) -> str:
    """Pick the install_only CPython 3.14 asset for this OS. Linux uses x86-64-v3."""
    if platform == "linux":
        suffix = "x86_64_v3-unknown-linux-gnu-install_only.tar.gz"
    elif platform == "windows":
        suffix = "x86_64-pc-windows-msvc-install_only.tar.gz"
    else:
        raise ValueError(f"unsupported platform {platform}")
    matches = [
        name
        for name in names
        if name.startswith("cpython-3.14.") and name.endswith(suffix) and "freethreaded" not in name
    ]
    if not matches:
        raise RuntimeError(f"no CPython 3.14 asset for {platform}")
    return sorted(matches)[-1]


def _release_names() -> list[str]:
    request = urllib.request.Request(
        "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest",
        headers={"User-Agent": "voxprint-enlarge", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - GitHub API
        payload = json.loads(response.read().decode("utf-8"))
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(assets, list):
        raise RuntimeError("the Python release list was empty")
    names = [str(item.get("name")) for item in assets if isinstance(item, dict) and item.get("name")]
    tag = str(payload.get("tag_name") or "")
    return [f"{tag}/{name}" if tag else name for name in names]


def install_python(dest: Path, platform: str) -> Path:
    """Download relocatable CPython into ``dest/python`` and return its interpreter."""
    tagged = _release_names()
    names = [item.split("/", 1)[-1] for item in tagged]
    asset = python_asset_name(names, platform)
    tag = ""
    for item in tagged:
        if item.endswith("/" + asset) or item == asset:
            tag = item.split("/", 1)[0] if "/" in item else ""
            break
    if not tag:
        raise RuntimeError("could not resolve the CPython release tag")
    url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{tag}/{asset}"
    archive = dest / asset
    urllib.request.urlretrieve(url, archive)  # noqa: S310 - the URL is the release above
    with tarfile.open(archive) as handle:
        handle.extractall(dest, filter="data")
    archive.unlink()
    if platform == "windows":
        return dest / "python" / "python.exe"
    return dest / "python" / "bin" / "python3"


def install_requirements(python: Path, requirements: Path) -> None:
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--platform", choices=("linux", "windows"), default="linux")
    parser.add_argument("--with-python", action="store_true")
    args = parser.parse_args(argv)
    stage_app(args.dest)
    stamp_build_info(args.dest)
    if args.with_python:
        python = install_python(args.dest, args.platform)
        install_requirements(python, ROOT / "requirements.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
