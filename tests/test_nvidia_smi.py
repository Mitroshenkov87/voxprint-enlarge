"""nvidia-smi is resolved to System32 before PATH, including from a patched probe."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from voxprint_engine.nvidia_smi import query_gpus, run_smi, smi_executable


def test_smi_executable_prefers_system32(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "Windows"
    exe = root / "System32" / "nvidia-smi.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("SystemRoot", str(root))
    assert smi_executable() == str(exe)


def test_smi_executable_falls_back_to_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("SystemRoot", str(tmp_path / "missing"))
    assert smi_executable() == "nvidia-smi"


def test_run_smi_rewrites_the_bare_name(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("voxprint_engine.nvidia_smi.subprocess.run", fake_run)
    monkeypatch.setattr(
        "voxprint_engine.nvidia_smi.smi_executable",
        lambda: r"C:\Windows\System32\nvidia-smi.exe",
    )
    run_smi(["nvidia-smi", "-L"], capture_output=True, text=True)
    assert seen["argv"] == [r"C:\Windows\System32\nvidia-smi.exe", "-L"]


def test_query_gpus_calls_the_probe_with_the_rewriting_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def fake(runner: object = None) -> list[object]:
        seen.append(runner)
        return []

    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", fake)
    assert query_gpus() == []
    assert seen == [run_smi]
