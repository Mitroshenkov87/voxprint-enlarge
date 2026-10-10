"""Startup gate and the compute-capability warning.

Enlarge starts only when nvidia-smi reports an NVIDIA GPU on a driver from the
600 series or newer, so CUDA can use it. A card below compute capability 8.9
still starts, and the warning is the only extra signal. ``VOXPRINT_CI_LANCZOS=1``
skips the gate and is the only way a job resamples with Lanczos. That flag is
for tests and CI. It is not a user setting.

The x86-64 CPU baseline is x86-64-v3, which includes AVX2 and does not include
AVX-512. Nothing here probes CPU flags: the same Python path runs on ARM, and
a missing x86 feature is not a fault.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Callable

MIN_COMPUTE = (8, 9)
MIN_DRIVER_MAJOR = 600
NO_GPU_MESSAGE = (
    "Enlarge needs an NVIDIA GPU that CUDA can use. "
    "This computer does not have one, so Enlarge will not start."
)
OLD_DRIVER_MESSAGE = (
    "Enlarge needs an NVIDIA driver from the 600 series or newer. "
    "This computer has driver {driver}. Update the driver, then start Enlarge again."
)


def parse_compute_capability(text: str, index: int = 0) -> tuple[int, int] | None:
    """Parse ``nvidia-smi --query-gpu=compute_cap`` CSV. ``None`` when the text has no number."""
    rows: list[tuple[int, int]] = []
    for line in text.splitlines():
        cell = line.split(",")[0].strip()
        major_text, dot, minor_text = cell.partition(".")
        if not dot:
            continue
        try:
            rows.append((int(major_text), int(minor_text)))
        except ValueError:
            continue
    if not rows:
        return None
    if index < 0 or index >= len(rows):
        index = 0
    return rows[index]


def capability_warning(cap: tuple[int, int] | None) -> str:
    """A short warning when the card is below 8.9. An unknown card returns an empty string."""
    if cap is None or cap >= MIN_COMPUTE:
        return ""
    major, minor = cap
    return (
        f"GPU compute capability {major}.{minor} is below 8.9 (RTX 40 / Ada). "
        "Enlarge is built for RTX 40 and newer."
    )


def ci_lanczos() -> bool:
    """True only for the internal test flag. Users cannot turn this on from the app."""
    return os.environ.get("VOXPRINT_CI_LANCZOS", "").strip() == "1"


def driver_major(version: str) -> int | None:
    """``610.73.01`` -> 610. ``None`` when the text is not a driver version."""
    head = version.strip().split(".", 1)[0]
    if not head.isdigit():
        return None
    return int(head)


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver: str
    compute: tuple[int, int] | None


@dataclass(frozen=True)
class LaunchGate:
    ok: bool
    message: str
    warning: str


def parse_gpu_table(text: str) -> list[GpuInfo]:
    """Parse ``name, driver_version, compute_cap`` CSV from nvidia-smi."""
    rows: list[GpuInfo] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name = ", ".join(parts[:-2]).strip()
        driver = parts[-2].strip()
        if not name or not driver or driver.upper() == "N/A":
            continue
        cap_text = parts[-1]
        major_text, dot, minor_text = cap_text.partition(".")
        compute: tuple[int, int] | None
        try:
            compute = (int(major_text), int(minor_text)) if dot else None
        except ValueError:
            compute = None
        rows.append(GpuInfo(name, driver, compute))
    return rows


def query_gpus(runner: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> list[GpuInfo]:
    """Ask nvidia-smi for every GPU. An empty list means CUDA cannot see a device."""
    run = runner or subprocess.run
    command = ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]
    try:
        if os.name == "nt":
            proc = run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        else:
            proc = run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return parse_gpu_table(proc.stdout or "")


def assess_gpu(index: int = 0, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> LaunchGate:
    """Whether Enlarge may start, plus the compute-capability warning when it may."""
    if ci_lanczos():
        return LaunchGate(True, "", "")
    gpus = query_gpus(runner)
    if not gpus:
        return LaunchGate(False, NO_GPU_MESSAGE, "")
    if index < 0 or index >= len(gpus):
        index = 0
    gpu = gpus[index]
    major = driver_major(gpu.driver)
    if major is None or major < MIN_DRIVER_MAJOR:
        shown = gpu.driver or "unknown"
        return LaunchGate(False, OLD_DRIVER_MESSAGE.format(driver=shown), "")
    return LaunchGate(True, "", capability_warning(gpu.compute))


def query_compute_capability(
    index: int = 0,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[int, int] | None:
    """Ask nvidia-smi. ``None`` when the tool is missing or the GPU does not report a capability."""
    run = runner or subprocess.run
    command = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    try:
        if os.name == "nt":
            proc = run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        else:
            proc = run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return parse_compute_capability(proc.stdout or "", index)
