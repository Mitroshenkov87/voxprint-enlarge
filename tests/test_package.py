"""Release staging: no weights, build numbers, and the installer contract."""
from __future__ import annotations

import json
from pathlib import Path

from enlarge.ipc.register import unregister_linux, write_linux
from tools.assemble_bundle import python_asset_name, release_version, stage_app
from tools.build_number import build_number, codename
from tools.generate_cli_reference import generate, render


def test_build_number_uses_the_run_and_the_offset() -> None:
    assert build_number({}) is None
    assert build_number({"GITHUB_RUN_NUMBER": "4"}) == 4
    assert build_number({"VOXPRINT_BUILD": "9", "GITHUB_RUN_NUMBER": "4"}) == 9
    assert codename() == "Emunah"
    assert release_version() == "0.2.0-rc.6"
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
    assert "/opt/voxprint-ai-suite/enlarge" in text
    assert "VOXPrint AI Enlarge" in text
    assert "SUDO_USER" in text
    assert "*/voxprint/enlarge" in text
    uninstall = (dest / "uninstall.sh").read_text(encoding="utf-8")
    assert "--uninstall-suite" in uninstall
    assert "--unregister" in uninstall
    assert "/opt/voxprint-ai-suite/enlarge" in uninstall
    launcher = (dest / "voxprint-enlarge").read_text(encoding="utf-8")
    assert "voxprint/shared/python/bin/python3" in launcher
    assert "import voxprint_common, voxprint_engine, onnxruntime" in launcher
    weights = [path for path in dest.rglob("*") if path.suffix.lower() in {".pth", ".onnx"}]
    assert weights == []
    iss = Path("installer/windows/enlarge.iss").read_text(encoding="utf-8")
    assert r"{autopf}\VOXPrint AI Suite\Enlarge" in iss
    assert "PrivilegesRequired=admin" in iss
    assert "voxprint-enlarge-setup" in iss
    assert "voxprint-enlarge.exe" in iss
    assert "voxprint-enlarge-cli.exe" in iss
    assert r"{autoprograms}\VOXPrint AI Suite\VOXPrint AI Enlarge" in iss
    migrate = iss.split("procedure MigratePerUserInstall", 1)[1].split("procedure CurStepChanged", 1)[0]
    assert r"{localappdata}\Voxprint\Enlarge" in migrate
    assert "DelTree" in migrate
    assert r"{userprograms}\Voxprint\Voxprint AI Enlarge.lnk" in migrate
    assert "--uninstall-suite" not in migrate
    assert "unins000" not in iss
    assert "function InitializeSetup" in iss
    assert r"Uninstall\{A7E1C4B2-6D40-4F1A-9C3E-8B5D2F0E1A44}_is1" in iss
    assert "--uninstall-suite" in iss
    assert "--unregister" in iss
    assert "--install-suite" in iss
    assert "--register" in iss
    launcher = Path("tools/launcher/Program.cs").read_text(encoding="utf-8")
    assert "#if GUI_LAUNCHER" in launcher
    gui_project = Path("tools/launcher/VoxprintEnlarge.csproj").read_text(encoding="utf-8")
    cli_project = Path("tools/launcher/voxprint-enlarge-cli.csproj").read_text(encoding="utf-8")
    assert "<AssemblyName>voxprint-enlarge</AssemblyName>" in gui_project
    assert "<OutputType>WinExe</OutputType>" in gui_project
    assert "GUI_LAUNCHER" in gui_project
    assert "<AssemblyName>voxprint-enlarge-cli</AssemblyName>" in cli_project
    assert "<OutputType>Exe</OutputType>" in cli_project
    assert "GUI_LAUNCHER" not in cli_project
    assert 'Path.Combine(local, "Voxprint", "shared")' in launcher
    assert "interpreter.json" in launcher
    assert "import voxprint_common, voxprint_engine, onnxruntime" in launcher
    assert "GetConsoleMode" in launcher
    assert "StdioRedirected" in launcher
    assert ".enlarge-import-probe" in launcher
    assert "--install-suite" in launcher
    assert "python.exe" in iss
    assert "Abort" in iss
    assert "enlarge-install.log" in iss
    assert "SuppressibleMsgBox" in iss
    assert "WizardSilent" in iss
    assert "MsgBox(" not in iss.replace("SuppressibleMsgBox(", "")


def test_unregister_removes_the_linux_discovery_files(tmp_path: Path) -> None:
    exe = tmp_path / "voxprint-enlarge"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    write_linux(exe, tmp_path / "share", tmp_path / "bin", ipc_name="/tmp/enlarge.sock")
    assert (tmp_path / "share" / "applications" / "ai.voxprint.Enlarge.desktop").is_file()
    unregister_linux(tmp_path / "share", tmp_path / "bin")
    assert not (tmp_path / "share" / "applications" / "ai.voxprint.Enlarge.desktop").exists()
    assert not (tmp_path / "bin" / "voxprint-enlarge").exists()
