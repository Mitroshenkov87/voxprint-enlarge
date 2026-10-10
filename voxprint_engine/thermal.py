"""Suite thermal rule, as a pure decision.

Full speed for the first 2.5 hours. After that, pause between batches while the
GPU's 5-minute median is at least 83 C, or while it has been throttling for
more than 60 seconds. Resume at or below 75 C.

The worker does not pause between batches yet. The function is here so Enlarge
and Clarity share one rule. Audiobook and Movie Dubber still have their own
pauses until that shared module replaces them.
"""
from __future__ import annotations

from dataclasses import dataclass

FULL_SPEED_S = 2.5 * 60 * 60
PAUSE_C = 83.0
RESUME_C = 75.0
THROTTLE_S = 60.0


@dataclass(frozen=True)
class ThermalDecision:
    """``action`` is ``run``, ``pause``, or ``resume``."""

    action: str
    reason: str


def decide(elapsed_s: float, median_c: float | None, throttling_s: float, paused: bool) -> ThermalDecision:
    """What to do before the next batch.

    ``median_c`` None means the sensor could not be read. A missing sensor does
    not by itself pause the run; a long throttle reading still does, once the
    full-speed window is over.
    """
    elapsed = max(0.0, float(elapsed_s))
    throttle = max(0.0, float(throttling_s))
    if elapsed < FULL_SPEED_S:
        return ThermalDecision("run", "full speed for the first 2.5 hours")
    hot = median_c is not None and median_c >= PAUSE_C
    throttling = throttle > THROTTLE_S
    if paused:
        if median_c is not None and median_c <= RESUME_C and not throttling:
            return ThermalDecision("resume", "GPU at or below 75 C")
        if hot:
            return ThermalDecision("pause", "5-minute median is at least 83 C")
        if throttling:
            return ThermalDecision("pause", "GPU has been throttling for more than 60 seconds")
        return ThermalDecision("pause", "waiting until the GPU is at or below 75 C")
    if hot:
        return ThermalDecision("pause", "5-minute median is at least 83 C")
    if throttling:
        return ThermalDecision("pause", "GPU has been throttling for more than 60 seconds")
    return ThermalDecision("run", "within limits")
