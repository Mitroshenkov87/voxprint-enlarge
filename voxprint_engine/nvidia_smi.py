"""Locate ``nvidia-smi`` for sessions that do not search ``System32``.

A scheduled task started without a console often has a ``PATH`` that omits
``System32``. ``query_gpus`` in ``voxprint_common`` runs ``nvidia-smi`` by
name, so this module rewrites that command to ``%SystemRoot%\\System32\\nvidia-smi.exe``
when the file is there, then falls back to ``PATH``.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

from voxprint_common import gpu_probe
from voxprint_common.gpu_probe import GpuDevice, Probe

_SMI_NAMES = {"nvidia-smi", "nvidia-smi.exe"}


def smi_executable() -> str:
    """Absolute ``nvidia-smi`` on Windows when System32 has it, otherwise the bare name."""
    if os.name == "nt":
        root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or r"C:\Windows"
        candidate = os.path.join(root, "System32", "nvidia-smi.exe")
        if os.path.isfile(candidate):
            return candidate
    return "nvidia-smi"


def smi_command(args: Sequence[str]) -> list[str]:
    """The same arguments, with the executable replaced by :func:`smi_executable`."""
    argv = [str(part) for part in args]
    if not argv:
        return [smi_executable()]
    if os.path.basename(argv[0]).lower() in _SMI_NAMES:
        return [smi_executable(), *argv[1:]]
    return [smi_executable(), *argv]


def run_smi(
    command: Sequence[str],
    *,
    capture_output: bool = False,
    text: bool = False,
    timeout: float | None = None,
    check: bool = False,
    creationflags: int = 0,
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run`` for an ``nvidia-smi`` command, using the resolved executable."""
    argv = smi_command(command)
    if creationflags:
        completed = subprocess.run(
            argv,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=check,
            creationflags=creationflags,
        )
    else:
        completed = subprocess.run(
            argv,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=check,
        )
    return completed


def query_gpus() -> list[GpuDevice]:
    """GPUs from ``voxprint_common``, with ``nvidia-smi`` resolved to a full path.

    The call goes through ``gpu_probe.query_gpus`` at call time so a test can
    replace that function and still be the one that runs.
    """
    return list(gpu_probe.query_gpus(runner=run_smi))


def assess(
    app: str,
    *,
    index: int = 0,
    runner: object = None,
    refuse_below: tuple[int, int] | None = None,
    devices: tuple[GpuDevice, ...] | None = None,
) -> Probe:
    """``voxprint_common.gpu_probe.assess`` using the resolved ``nvidia-smi``.

    A caller that passes ``runner`` or ``devices`` keeps that input. Production
    calls pass neither, so the query uses :func:`query_gpus`.
    """
    if runner is None and devices is None:
        devices = tuple(query_gpus())
    return gpu_probe.assess(app, index=index, runner=runner, refuse_below=refuse_below, devices=devices)  # type: ignore[arg-type]
