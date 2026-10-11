"""Startup gate and the compute-capability warning."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from voxprint_engine.hardware import (
    NO_GPU_MESSAGE,
    OLD_DRIVER_MESSAGE,
    assess_gpu,
    capability_warning,
    driver_major,
    parse_compute_capability,
    parse_gpu_table,
    query_compute_capability,
)


def test_parse_and_warn() -> None:
    assert parse_compute_capability("8.9\n") == (8, 9)
    assert parse_compute_capability("8.6\n9.0\n", index=1) == (9, 0)
    assert parse_compute_capability("N/A\n") is None
    assert parse_compute_capability("") is None
    assert capability_warning(None) == ""
    assert capability_warning((8, 9)) == ""
    assert capability_warning((9, 0)) == ""
    warning = capability_warning((8, 6))
    assert "8.6" in warning
    assert "8.9" in warning
    assert "RTX 40" in warning


def test_query_uses_nvidia_smi_and_ignores_a_missing_tool() -> None:
    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout="8.6\n", stderr="")

    assert query_compute_capability(0, runner) == (8, 6)

    def missing(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("not found")

    assert query_compute_capability(0, missing) is None


def _smi(stdout: str, code: int = 0):
    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=code, stdout=stdout, stderr="")

    return runner


def test_driver_and_gpu_table() -> None:
    assert driver_major("610.73.01") == 610
    assert driver_major("560.35.03") == 560
    assert driver_major("n/a") is None
    rows = parse_gpu_table("NVIDIA GeForce RTX 3090, 610.73.01, 8.6\n")
    assert len(rows) == 1
    assert rows[0].name == "NVIDIA GeForce RTX 3090"
    assert rows[0].driver == "610.73.01"
    assert rows[0].compute == (8, 6)
    assert parse_gpu_table("") == []


def test_gate_refuses_when_cuda_cannot_see_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    gate = assess_gpu(runner=_smi("", code=1))
    assert gate.ok is False
    assert gate.message == NO_GPU_MESSAGE.format(app_name="This program")
    assert gate.warning == ""


def test_gate_refuses_a_driver_below_600(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    gate = assess_gpu(runner=_smi("NVIDIA GeForce RTX 4090, 580.88, 8.9\n"))
    assert gate.ok is False
    assert gate.message == OLD_DRIVER_MESSAGE.format(app_name="This program", driver="580.88")
    assert gate.warning == ""


def test_gate_warns_below_8_9_and_still_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    gate = assess_gpu(runner=_smi("NVIDIA GeForce RTX 3090, 610.73.01, 8.6\n"))
    assert gate.ok is True
    assert gate.message == ""
    assert "8.6" in gate.warning


def test_gate_is_quiet_on_ada_with_a_600_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    gate = assess_gpu(runner=_smi("NVIDIA GeForce RTX 4090, 610.73.01, 8.9\n"))
    assert gate.ok is True
    assert gate.message == ""
    assert gate.warning == ""


def test_gpu_refusal_uses_the_caller_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    missing = assess_gpu(runner=_smi("", code=1), app_name="Voxprint AI Canvas")
    assert missing.message == NO_GPU_MESSAGE.format(app_name="Voxprint AI Canvas")
    assert "Enlarge" not in missing.message
    old = assess_gpu(runner=_smi("NVIDIA GeForce RTX 4090, 580.88, 8.9\n"), app_name="Voxprint AI Canvas")
    assert old.message == OLD_DRIVER_MESSAGE.format(app_name="Voxprint AI Canvas", driver="580.88")
    warned = assess_gpu(runner=_smi("NVIDIA GeForce RTX 3090, 610.73.01, 8.6\n"), app_name="Voxprint AI Canvas")
    assert "Voxprint AI Canvas is built for RTX 40" in warned.warning
    root = Path(__file__).resolve().parents[1]
    hardware = (root / "voxprint_engine" / "hardware.py").read_text(encoding="utf-8")
    infer = (root / "voxprint_engine" / "infer.py").read_text(encoding="utf-8")
    assert "Voxprint AI Enlarge" not in hardware
    assert "Enlarge needs" not in hardware
    assert "Enlarge needs" not in infer


def test_ci_flag_skips_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_CI_LANCZOS", "1")

    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("nvidia-smi must not run")

    gate = assess_gpu(runner=runner)
    assert gate.ok is True
    assert gate.message == ""
    assert gate.warning == ""


def test_runtime_pins() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/audit.yml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.14,<3.15"' in pyproject
    assert "imagecodecs==2026.10.10" in requirements
    assert 'python-version: "3.14"' in workflow
    assert "600 series" in readme
    assert "does not start" in readme
