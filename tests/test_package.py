"""Release staging: no weights, build numbers, and the installer contract."""
from __future__ import annotations

from pathlib import Path

from enlarge.ipc.register import unregister_linux, write_linux
from tools.assemble_bundle import python_asset_name, stage_app
from tools.build_number import build_number, codename


def test_build_number_uses_the_run_and_the_offset() -> None:
    assert build_number({}) == 0
    assert build_number({"GITHUB_RUN_NUMBER": "4"}) == 4
    assert build_number({"VOXPRINT_BUILD": "9", "GITHUB_RUN_NUMBER": "4"}) == 9
    assert codename() == "Hagdil"


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
    assert "--register" in text
    assert "--unregister" in (dest / "uninstall.sh").read_text(encoding="utf-8")
    weights = [path for path in dest.rglob("*") if path.suffix.lower() in {".pth", ".onnx"}]
    assert weights == []
    iss = Path("installer/windows/enlarge.iss").read_text(encoding="utf-8")
    assert r"{localappdata}\Voxprint\Enlarge" in iss
    assert r"{userprograms}\Voxprint" in iss
    assert "--unregister" in iss
    assert "--register" in iss


def test_unregister_removes_the_linux_discovery_files(tmp_path: Path) -> None:
    exe = tmp_path / "voxprint-enlarge"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    write_linux(exe, tmp_path / "share", tmp_path / "bin", ipc_name="/tmp/enlarge.sock")
    assert (tmp_path / "share" / "applications" / "ai.voxprint.Enlarge.desktop").is_file()
    unregister_linux(tmp_path / "share", tmp_path / "bin")
    assert not (tmp_path / "share" / "applications" / "ai.voxprint.Enlarge.desktop").exists()
    assert not (tmp_path / "bin" / "voxprint-enlarge").exists()
