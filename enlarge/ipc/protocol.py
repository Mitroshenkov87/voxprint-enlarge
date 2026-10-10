"""JSON-line messages for protocol 1.

Every object carries ``protocol``. Readers ignore keys they do not know.
A major change bumps ``protocol``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, assert_never

from voxprint_engine.formats import DEFAULT_QUALITY

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
    """IPC-ENLARGE retry table. ``gpu_busy`` and ``model_missing`` can be tried again."""
    if code == "gpu_busy" or code == "model_missing":
        return True
    if (
        code == "gpu_unsupported"
        or code == "oom"
        or code == "bad_input"
        or code == "unsupported_format"
        or code == "io_error"
        or code == "cancelled"
        or code == "internal"
    ):
        return False
    assert_never(code)


def exit_code(code: ErrorCode) -> int:
    """Process status for one error code. Success is 0 and is not a code."""
    if code == "internal":
        return 1
    if code == "bad_input":
        return 2
    if code == "unsupported_format":
        return 3
    if code == "gpu_unsupported":
        return 4
    if code == "gpu_busy":
        return 5
    if code == "oom":
        return 6
    if code == "model_missing":
        return 7
    if code == "io_error":
        return 8
    if code == "cancelled":
        return 130
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


def _kind(name: str) -> dict[str, str]:
    """``op`` keeps published readers working. ``type`` is the IPC-ENLARGE name."""
    return {"op": name, "type": name}


def hello_message(version: str) -> dict[str, Any]:
    """Server hello: version, protocol range, and both op and type."""
    return {
        **_kind("hello"),
        "version": version,
        "protocol_min": PROTOCOL_MIN,
        "protocol_max": PROTOCOL_MAX,
        "server": "enlarge",
        "protocols": [PROTOCOL],
    }


def accepted_message(job_id: str, queue_pos: int) -> dict[str, Any]:
    """The job is queued. queue_pos is how many jobs were already ahead."""
    return {**_kind("accepted"), "id": job_id, "queue_pos": queue_pos}


def progress_message(
    job_id: str,
    pct: float,
    stage: Stage,
    eta_s: float | None = None,
    lock_holder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Progress in percent. waiting_gpu also carries the lock holder."""
    bounded = max(0.0, min(100.0, float(pct)))
    eta = None if eta_s is None else round(max(0.0, float(eta_s)), 1)
    body: dict[str, Any] = {
        **_kind("progress"),
        "id": job_id,
        "pct": round(bounded, 1),
        "stage": stage,
        "eta_s": eta,
    }
    if stage == "waiting_gpu" and lock_holder is not None:
        body["lock_holder"] = {
            "owner": lock_holder.get("owner"),
            "pid": lock_holder.get("pid"),
            "job": lock_holder.get("job"),
        }
    return body


def done_message(job_id: str, output: str, width: int, height: int, image_format: str, ms: float) -> dict[str, Any]:
    """The finished file, its pixel size, the format, and the elapsed milliseconds."""
    return {
        **_kind("done"),
        "id": job_id,
        "output": output,
        "width": width,
        "height": height,
        "format": image_format,
        "ms": round(ms, 1),
    }


def error_message(job_id: str, code: ErrorCode, message: str) -> dict[str, Any]:
    """An error object. retryable is true only for gpu_busy and model_missing."""
    return {
        **_kind("error"),
        "id": job_id,
        "code": code,
        "message": message,
        "retryable": is_retryable(code),
    }


def message_kind(message: dict[str, Any]) -> str:
    """``type`` when a client sends the IPC-ENLARGE name, otherwise ``op``."""
    kind = message.get("type")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    kind = message.get("op")
    if isinstance(kind, str):
        return kind.strip()
    return ""


@dataclass(frozen=True)
class EnlargeRequest:
    """One enlarge.request. Paths only; the socket never carries image bytes."""
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
    lossless: bool = False
    encode_quality: int = DEFAULT_QUALITY
    long_side: int = 0
    lock_timeout: float = 2.0
    parent_lock_pid: int = 0


def _text(message: dict[str, Any], key: str) -> str:
    value = message.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip()


def _whole(message: dict[str, Any], key: str, default: int) -> int:
    value = message.get(key, default)
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


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
    preset = _text(message, "preset")
    model = _text(message, "model")
    quality = message.get("quality", "fast")
    if quality is None or quality == "":
        quality = "fast"
    if not isinstance(quality, str):
        raise ValueError("quality must be fast or max")
    image_format = _text(message, "format")
    output = _text(message, "output")
    output_dir = _text(message, "output_dir")
    encode_quality = _whole(message, "encode_quality", DEFAULT_QUALITY)
    long_side = _whole(message, "long_side", 0)
    parent_lock_pid = _whole(message, "parent_lock_pid", 0)
    timeout = message.get("lock_timeout", 2.0)
    if timeout is None or timeout == "":
        timeout_s = 2.0
    elif isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("lock_timeout must be a number of seconds")
    else:
        timeout_s = float(timeout)
    return EnlargeRequest(
        id=job_id.strip(),
        input=source.strip(),
        scale=scale,
        preset=preset.lower(),
        model=model,
        quality=quality.strip().lower(),
        format=image_format.lower(),
        output=output,
        output_dir=output_dir,
        open_ui=message.get("open_ui") is True,
        lossless=message.get("lossless") is True,
        encode_quality=encode_quality,
        long_side=long_side,
        lock_timeout=timeout_s,
        parent_lock_pid=parent_lock_pid,
    )
