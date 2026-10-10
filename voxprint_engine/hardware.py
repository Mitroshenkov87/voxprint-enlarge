"""Diagnostics for the suite hardware floor. Warnings do not stop a job.

The GPU floor is NVIDIA RTX 40 (Ada, compute capability 8.9) and newer.
The x86-64 CPU baseline is x86-64-v3, which includes AVX2 and does not include
AVX-512. Nothing here probes CPU flags: the same Python path runs on ARM, and
a missing x86 feature is not a fault.
"""
from __future__ import annotations

import os
import subprocess
from typing import Callable

MIN_COMPUTE = (8, 9)


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
        "Enlarge is built for RTX 40 and newer. Lanczos can still preview."
    )


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
