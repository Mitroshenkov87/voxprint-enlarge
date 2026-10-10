"""VRAM budget shared by Enlarge and Clarity.

Suite rule: ``budget = free_now - max(2 GB, 8% of total)``, measured again
before each batch. GB here is 1024^3, matching the other Voxprint programs.
An optional user fraction is not applied in phase 1; the headroom is the whole margin.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Callable

GB = 1024 ** 3
HEADROOM_GB = 2.0
HEADROOM_FRAC = 0.08
# Tiles the ONNX tiler will use. The Lanczos preview reports the same choice
# and does not actually cut the picture up.
TILES = (128, 256, 512, 768)


@dataclass(frozen=True)
class VramSnapshot:
    device: str
    total_gb: float
    free_gb: float
    budget_gb: float
    source: str


def headroom_gb(total_gb: float) -> float:
    """Memory left unused on purpose: at least 2 GB, or 8% of the card when that is larger."""
    return max(HEADROOM_GB, HEADROOM_FRAC * max(0.0, total_gb))


def vram_budget_gb(free_gb: float, total_gb: float) -> float:
    """``max(0, free_now - max(2 GB, 8% of total))``."""
    return max(0.0, float(free_gb) - headroom_gb(float(total_gb)))


def note_spill(batch: int, spilled: bool) -> int:
    """Halve the batch when work spills into system RAM. The batch never drops below 1."""
    size = max(1, int(batch))
    if spilled:
        return max(1, size // 2)
    return size


def auto_tile(budget_gb: float) -> int:
    """Pick a tile from the memory that is actually free to use."""
    if budget_gb < 1.5:
        return 128
    if budget_gb < 4.0:
        return 256
    if budget_gb < 8.0:
        return 512
    return 768


def choose_tile(budget_gb: float, model_tile: int = 0, user_tile: int = 0) -> int:
    """``user_tile`` wins when it is set. Otherwise the model's hint is clamped to the budget ladder."""
    if user_tile:
        tile = int(user_tile)
        if 32 <= tile <= 2048:
            return tile
    auto = auto_tile(budget_gb)
    if model_tile and model_tile > 0:
        return min(int(model_tile), auto)
    return auto


def overlap_for(tile: int) -> int:
    """Feather (pixels) the real tiler will overlap. Reported now so the UI can show the plan."""
    if tile < 64:
        return 0
    return 16 if tile >= 128 else 8


def parse_nvidia_smi(text: str, index: int = 0) -> tuple[float, float] | None:
    """Parse ``memory.total, memory.free`` CSV in MiB. Returns ``(total_gb, free_gb)``."""
    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            total_mib = float(parts[0])
            free_mib = float(parts[1])
        except ValueError:
            continue
        rows.append((total_mib / 1024.0, free_mib / 1024.0))
    if not rows:
        return None
    if index < 0 or index >= len(rows):
        index = 0
    return rows[index]


def query_nvidia_smi(index: int = 0, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> tuple[float, float] | None:
    """Ask nvidia-smi. ``None`` when the tool is missing or returns nothing useful."""
    run = runner or subprocess.run
    command = ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"]
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
    return parse_nvidia_smi(proc.stdout or "", index)


def device_index(device: str) -> int:
    """``cuda:N`` -> N. Anything else is GPU 0, which nvidia-smi uses when the user left the choice on auto."""
    text = (device or "").strip().lower()
    if text.startswith("cuda:"):
        try:
            return max(0, int(text.split(":", 1)[1]))
        except ValueError:
            return 0
    return 0


def measure(device: str = "auto") -> VramSnapshot:
    """Budget for ``device``. A CPU choice, or a machine with no nvidia-smi, reports a zero GPU budget."""
    if (device or "").strip().lower() == "cpu":
        return VramSnapshot("cpu", 0.0, 0.0, 0.0, "cpu")
    found = query_nvidia_smi(device_index(device))
    if found is None:
        return VramSnapshot(device or "auto", 0.0, 0.0, 0.0, "none")
    total, free = found
    return VramSnapshot(device or "auto", total, free, vram_budget_gb(free, total), "nvidia-smi")
