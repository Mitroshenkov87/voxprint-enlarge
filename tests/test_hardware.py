"""Compute-capability warning. It reports old cards and does not refuse them."""
from __future__ import annotations

import subprocess

from voxprint_engine.hardware import capability_warning, parse_compute_capability, query_compute_capability


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
    assert "Lanczos" in warning


def test_query_uses_nvidia_smi_and_ignores_a_missing_tool() -> None:
    def runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout="8.6\n", stderr="")

    assert query_compute_capability(0, runner) == (8, 6)

    def missing(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("not found")

    assert query_compute_capability(0, missing) is None
