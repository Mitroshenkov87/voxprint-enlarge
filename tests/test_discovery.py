"""Install-path discovery: the suite location, then a legacy record, then PATH."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from enlarge.ipc.discovery import legacy_install_exe, resolve_exe
from enlarge.ipc.register import gui_exe_for


def test_resolve_exe_prefers_the_suite_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = tmp_path / "VOXPrint AI Suite" / "Enlarge" / "voxprint-enlarge-cli.exe"
    suite.parent.mkdir(parents=True)
    suite.write_text("new", encoding="utf-8")
    legacy = tmp_path / "Voxprint" / "Enlarge" / "VoxprintEnlarge.exe"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("old", encoding="utf-8")
    monkeypatch.setattr("enlarge.ipc.discovery.known_install_exe", lambda: suite)
    monkeypatch.setattr("enlarge.ipc.discovery.legacy_install_exe", lambda: legacy)
    monkeypatch.setattr("enlarge.ipc.discovery._registry_exe", lambda: legacy)
    monkeypatch.setattr("enlarge.ipc.discovery.shutil.which", lambda _name: None)
    assert resolve_exe("missing") == suite.resolve()


def test_resolve_exe_reads_an_old_registry_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = tmp_path / "missing" / "voxprint-enlarge-cli.exe"
    recorded = tmp_path / "old" / "VoxprintEnlarge.exe"
    recorded.parent.mkdir(parents=True)
    recorded.write_text("old", encoding="utf-8")
    legacy = tmp_path / "other" / "VoxprintEnlarge.exe"
    monkeypatch.setattr("enlarge.ipc.discovery.known_install_exe", lambda: suite)
    monkeypatch.setattr("enlarge.ipc.discovery.legacy_install_exe", lambda: legacy)
    monkeypatch.setattr("enlarge.ipc.discovery._registry_exe", lambda: recorded)
    monkeypatch.setattr("enlarge.ipc.discovery.shutil.which", lambda _name: None)
    assert resolve_exe("missing") == recorded.resolve()


def test_resolve_exe_falls_back_to_the_legacy_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = tmp_path / "missing" / "voxprint-enlarge-cli.exe"
    legacy = tmp_path / "Voxprint" / "Enlarge" / "VoxprintEnlarge.exe"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("old", encoding="utf-8")
    monkeypatch.setattr("enlarge.ipc.discovery.known_install_exe", lambda: suite)
    monkeypatch.setattr("enlarge.ipc.discovery.legacy_install_exe", lambda: legacy)
    monkeypatch.setattr("enlarge.ipc.discovery._registry_exe", lambda: None)
    monkeypatch.setattr("enlarge.ipc.discovery.shutil.which", lambda _name: None)
    assert resolve_exe("missing") == legacy.resolve()


def test_legacy_install_exe_keeps_the_previous_leaf() -> None:
    path = legacy_install_exe()
    if sys.platform == "win32":
        assert path.name == "VoxprintEnlarge.exe"
        assert "Voxprint" in path.parts and "Enlarge" in path.parts
        return
    assert path.parts[-3:] == ("voxprint", "enlarge", "voxprint-enlarge")


def test_gui_exe_for_uses_the_window_launcher_beside_the_cli(tmp_path: Path) -> None:
    folder = tmp_path / "Enlarge"
    folder.mkdir()
    cli = folder / "voxprint-enlarge-cli.exe"
    gui = folder / "voxprint-enlarge.exe"
    cli.write_text("cli", encoding="utf-8")
    assert gui_exe_for(cli) == cli
    gui.write_text("gui", encoding="utf-8")
    assert gui_exe_for(cli) == gui
