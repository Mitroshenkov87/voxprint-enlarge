"""Release staging: no weights, build numbers, and the installer contract."""
from __future__ import annotations

import json
from pathlib import Path

from enlarge.ipc.register import unregister_linux, write_linux
from tools.assemble_bundle import python_asset_name, release_version, stage_app
from tools.build_number import build_number, codename
from tools.generate_cli_reference import generate, render


def test_build_number_uses_the_run_and_the_offset() -> None:
    assert build_number({}) == 0
    assert build_number({"GITHUB_RUN_NUMBER": "4"}) == 4
    assert build_number({"VOXPRINT_BUILD": "9", "GITHUB_RUN_NUMBER": "4"}) == 9
    assert codename() == "Marpe"
    assert release_version() == "0.2.0-rc.3"
    iss = Path("installer/windows/enlarge.iss").read_text(encoding="utf-8")
    assert f'#define AppVersion "{release_version()}"' in iss
    assert json.loads(Path("BUILD.json").read_text(encoding="utf-8"))["version"] == release_version()


def test_cli_reference_is_generated_from_the_parser(tmp_path: Path) -> None:
    text = render()
    assert "--preset" in text
    assert "photo" in text
    assert "fast" in text
    assert "600" in text
    written = generate(tmp_path)
    assert written.name == "cli-reference.md"
    assert "CLI contract" in written.read_text(encoding="utf-8")
    roadmap = (tmp_path / "roadmap.md").read_text(encoding="utf-8")
    assert "Next version" in roadmap
    assert "face-restore" in roadmap


def test_python_asset_is_cpython_314_without_freethreading() -> None:
    names = [
        "cpython-3.14.8+20261009-x86_64_v3-unknown-linux-gnu-install_only.tar.gz",
        "cpython-3.14.8+20261009-x86_64_v3-unknown-linux-gnu-freethreaded-install_only.tar.gz",
        "cpython-3.14.8+20261009-x86_64-pc-windows-msvc-install_only.tar.gz",
        "cpython-3.13.1+20261009-x86_64_v3-unknown-linux-gnu-install_only.tar.gz",
    ]
    assert python_asset_name(names, "linux").startswith("cpython-3.14.")
    assert "freethreaded" not in python_asset_name(names, "linux")
    assert python_asset_name(names, "linux").endswith("x86_64_v3-unknown-linux-gnu-install_only.tar.gz")
    assert python_asset_name(names, "windows").endswith("x86_64-pc-windows-msvc-install_only.tar.gz")


def test_stage_app_leaves_weights_out(tmp_path: Path) -> None:
    dest = tmp_path / "bundle"
    stage_app(dest)
    assert (dest / "main.py").is_file()
    assert (dest / "voxprint_engine" / "registry" / "catalog.json").is_file()
    assert (dest / "voxprint-enlarge").is_file()
    text = (dest / "install.sh").read_text(encoding="utf-8")
    assert "--install-suite" in text
    assert "--register" in text
    uninstall = (dest / "uninstall.sh").read_text(encoding="utf-8")
    assert "--uninstall-suite" in uninstall
    assert "--unregister" in uninstall
    launcher = (dest / "voxprint-enlarge").read_text(encoding="utf-8")
    assert "voxprint/shared/python/bin/python3" in launcher
    weights = [path for path in dest.rglob("*") if path.suffix.lower() in {".pth", ".onnx"}]
    assert weights == []
    iss = Path("installer/windows/enlarge.iss").read_text(encoding="utf-8")
    assert r"{localappdata}\Voxprint\Enlarge" in iss
    assert r"{userprograms}\Voxprint" in iss
    assert "--uninstall-suite" in iss
    assert "--unregister" in iss
    assert "--install-suite" in iss
    assert "--register" in iss
    launcher = Path("tools/launcher/Program.cs").read_text(encoding="utf-8")
    assert 'Path.Combine(local, "Voxprint", "shared", "python")' in launcher


def test_unregister_removes_the_linux_discovery_files(tmp_path: Path) -> None:
    exe = tmp_path / "voxprint-enlarge"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    write_linux(exe, tmp_path / "share", tmp_path / "bin", ipc_name="/tmp/enlarge.sock")
    assert (tmp_path / "share" / "applications" / "ai.voxprint.Enlarge.desktop").is_file()
    unregister_linux(tmp_path / "share", tmp_path / "bin")
    assert not (tmp_path / "share" / "applications" / "ai.voxprint.Enlarge.desktop").exists()
    assert not (tmp_path / "bin" / "voxprint-enlarge").exists()
