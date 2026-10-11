"""Startup gate and the compute-capability warning, via voxprint_common.gpu_probe."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from main import _gpu_block
from voxprint_common.gpu_probe import (
    assess,
    capability_warning,
    driver_major,
    no_gpu_message,
    old_driver_message,
    parse_compute_capability,
    parse_gpu_table,
)


def test_parse_and_warn() -> None:
    assert parse_compute_capability("8.9\n") == (8, 9)
    assert parse_compute_capability("8.6\n9.0\n", index=1) == (9, 0)
    assert parse_compute_capability("N/A\n") is None
    assert parse_compute_capability("") is None
    assert capability_warning("Enlarge", None) == ""
    assert capability_warning("Enlarge", (8, 9)) == ""
    assert capability_warning("Enlarge", (9, 0)) == ""
    warning = capability_warning("Enlarge", (8, 6))
    assert "8.6" in warning
    assert "8.9" in warning
    assert "RTX 40" in warning


def _smi(stdout: str, code: int = 0):
    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=code, stdout=stdout, stderr="")

    return runner


def test_driver_and_gpu_table() -> None:
    assert driver_major("610.73.01") == 610
    assert driver_major("560.35.03") == 560
    assert driver_major("n/a") is None
    rows = parse_gpu_table("NVIDIA GeForce RTX 3090, 610.73.01, 8.6, 24576\n")
    assert len(rows) == 1
    assert rows[0].name == "NVIDIA GeForce RTX 3090"
    assert rows[0].driver == "610.73.01"
    assert rows[0].compute == (8, 6)
    assert rows[0].memory_mib == 24576
    assert parse_gpu_table("") == []


def test_gate_refuses_when_cuda_cannot_see_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    gate = assess("This program", runner=_smi("", code=1))
    assert gate.ok is False
    assert gate.message == no_gpu_message("This program")
    assert gate.warning == ""


def test_gate_refuses_a_driver_below_600(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    gate = assess("This program", runner=_smi("NVIDIA GeForce RTX 4090, 580.88, 8.9, 24564\n"))
    assert gate.ok is False
    assert gate.message == old_driver_message("This program", "580.88")
    assert gate.warning == ""


def test_gate_warns_below_8_9_and_still_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    gate = assess("This program", runner=_smi("NVIDIA GeForce RTX 3090, 610.73.01, 8.6, 24576\n"))
    assert gate.ok is True
    assert gate.message == ""
    assert "8.6" in gate.warning


def test_gate_is_quiet_on_ada_with_a_600_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    gate = assess("This program", runner=_smi("NVIDIA GeForce RTX 4090, 610.73.01, 8.9, 24564\n"))
    assert gate.ok is True
    assert gate.message == ""
    assert gate.warning == ""


def test_gpu_refusal_uses_the_caller_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    missing = assess("Voxprint AI Canvas", runner=_smi("", code=1))
    assert missing.message == no_gpu_message("Voxprint AI Canvas")
    assert "Enlarge" not in missing.message
    old = assess("Voxprint AI Canvas", runner=_smi("NVIDIA GeForce RTX 4090, 580.88, 8.9, 24564\n"))
    assert old.message == old_driver_message("Voxprint AI Canvas", "580.88")
    warned = assess("Voxprint AI Canvas", runner=_smi("NVIDIA GeForce RTX 3090, 610.73.01, 8.6, 24576\n"))
    assert "Voxprint AI Canvas is built for RTX 40" in warned.warning
    root = Path(__file__).resolve().parents[1]
    infer = (root / "voxprint_engine" / "infer.py").read_text(encoding="utf-8")
    assert "Enlarge needs" not in infer
    assert not (root / "voxprint_engine" / "hardware.py").exists()


def test_ci_flag_skips_the_enlarge_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_CI_LANCZOS", "1")

    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("nvidia-smi must not run")

    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", runner)
    assert _gpu_block() == ""


def test_runtime_pins() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/audit.yml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.14,<3.15"' in pyproject
    assert "imagecodecs==2026.10.10" in requirements
    assert "voxprint-common" in requirements
    assert 'python-version: "3.14"' in workflow
    assert "600 series" in readme
    assert "does not start" in readme
