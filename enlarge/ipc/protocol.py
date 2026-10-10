"""JSON-line messages for protocol 1.

Every object carries ``protocol``. Readers ignore keys they do not know.
A major change bumps ``protocol``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, assert_never

PROTOCOL = 1
PROTOCOL_MIN = 1
PROTOCOL_MAX = 1

ErrorCode = Literal[
    "gpu_unsupported",
    "gpu_busy",
    "oom",
    "bad_input",
    "unsupported_format",
    "io_error",
    "cancelled",
    "model_missing",
    "internal",
]
Stage = Literal["queued", "waiting_gpu", "loading", "tiles", "encoding"]
STAGES: tuple[Stage, ...] = ("queued", "waiting_gpu", "loading", "tiles", "encoding")
ERROR_CODES: tuple[ErrorCode, ...] = (
    "gpu_unsupported",
    "gpu_busy",
    "oom",
    "bad_input",
    "unsupported_format",
    "io_error",
    "cancelled",
    "model_missing",
    "internal",
)


def is_retryable(code: ErrorCode) -> bool:
    if code == "gpu_busy" or code == "oom" or code == "io_error" or code == "internal":
        return True
    if (
        code == "gpu_unsupported"
        or code == "bad_input"
        or code == "unsupported_format"
        or code == "cancelled"
        or code == "model_missing"
    ):
        return False
    assert_never(code)


def dumps(message: dict[str, Any]) -> bytes:
    """One UTF-8 JSON object and a newline."""
    body = dict(message)
    body["protocol"] = PROTOCOL
    line = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return (line + "\n").encode("utf-8")


def loads(line: bytes | str) -> dict[str, Any]:
    """Parse one line. Extra keys stay in the dict so callers can ignore them."""
    text = line.decode("utf-8") if isinstance(line, bytes) else line
    data = json.loads(text.strip())
    if not isinstance(data, dict):
        raise ValueError("message must be a JSON object")
    protocol = data.get("protocol")
    if protocol != PROTOCOL:
        raise ValueError("protocol must be 1")
    return data


def hello_message(version: str) -> dict[str, Any]:
    return {
        "op": "hello",
        "version": version,
        "protocol_min": PROTOCOL_MIN,
        "protocol_max": PROTOCOL_MAX,
    }


def accepted_message(job_id: str, queue_pos: int) -> dict[str, Any]:
    return {"op": "accepted", "id": job_id, "queue_pos": queue_pos}


def progress_message(job_id: str, pct: float, stage: Stage) -> dict[str, Any]:
    bounded = max(0.0, min(100.0, float(pct)))
    return {"op": "progress", "id": job_id, "pct": round(bounded, 1), "stage": stage}


def done_message(job_id: str, output: str, width: int, height: int, image_format: str, ms: float) -> dict[str, Any]:
    return {
        "op": "done",
        "id": job_id,
        "output": output,
        "width": width,
        "height": height,
        "format": image_format,
        "ms": round(ms, 1),
    }


def error_message(job_id: str, code: ErrorCode, message: str) -> dict[str, Any]:
    return {
        "op": "error",
        "id": job_id,
        "code": code,
        "message": message,
        "retryable": is_retryable(code),
    }


@dataclass(frozen=True)
class EnlargeRequest:
    id: str
    input: str
    scale: int
    preset: str
    model: str
    quality: str
    format: str
    output: str
    output_dir: str
    open_ui: bool


def request_from_message(message: dict[str, Any]) -> EnlargeRequest:
    """Read a request. Unknown keys are ignored. Invalid shapes raise ``ValueError``."""
    job_id = message.get("id")
    source = message.get("input")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("id must be a string")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("input must be a file path")
    scale = message.get("scale")
    if isinstance(scale, bool) or not isinstance(scale, int):
        raise ValueError("scale must be 2 or 4")
    preset = message.get("preset", "")
    if preset is None:
        preset = ""
    if not isinstance(preset, str):
        raise ValueError("preset must be a string")
    model = message.get("model", "")
    if model is None:
        model = ""
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    quality = message.get("quality", "fast")
    if quality is None or quality == "":
        quality = "fast"
    if not isinstance(quality, str):
        raise ValueError("quality must be fast or max")
    image_format = message.get("format", "")
    if image_format is None:
        image_format = ""
    if not isinstance(image_format, str):
        raise ValueError("format must be a string")
    output = message.get("output", "")
    if output is None:
        output = ""
    if not isinstance(output, str):
        raise ValueError("output must be a file path")
    output_dir = message.get("output_dir", "")
    if output_dir is None:
        output_dir = ""
    if not isinstance(output_dir, str):
        raise ValueError("output_dir must be a directory path")
    return EnlargeRequest(
        id=job_id.strip(),
        input=source.strip(),
        scale=scale,
        preset=preset.strip().lower(),
        model=model.strip(),
        quality=quality.strip().lower(),
        format=image_format.strip().lower(),
        output=output.strip(),
        output_dir=output_dir.strip(),
        open_ui=message.get("open_ui") is True,
    )
