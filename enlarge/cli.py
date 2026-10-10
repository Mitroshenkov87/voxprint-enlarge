"""Headless contract other suite apps call. It does not open a window.

A warm ``--serve`` server for this user receives the job over protocol 1.
Otherwise the job runs in this process. Stdout is the same JSON lines either way.
``--dry-run`` stays in this process: it loads no model and does not touch the GPU.
"""
from __future__ import annotations

import os
import signal
import sys
from typing import Any

from enlarge.appinfo import APP_VERSION
from enlarge.ipc.capabilities import capabilities
from enlarge.ipc.client import exchange
from enlarge.ipc.discovery import resolve_exe
from enlarge.ipc.jobs import JobControl, run_job
from enlarge.ipc.protocol import ERROR_CODES, EnlargeRequest, dumps, error_message, exit_code, message_kind
from voxprint_engine.formats import DEFAULT_QUALITY

_USAGE = (
    "usage: voxprint-enlarge --input FILE --output FILE (--scale 2|4 | --long-side PX) "
    "[--format avif|jxl|webp|jpeg|heic] [--lossless] [--quality 1-100] "
    "[--preset photo|art] [--model ID] [--quality-tier fast|max] "
    "[--lock-timeout SEC] [--parent-lock-pid PID] [--dry-run] --json-progress"
)


def write_json(message: dict[str, Any]) -> None:
    data = dumps(message)
    stream = sys.stdout
    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        stream.write(data.decode("utf-8"))
        stream.flush()
        return
    buffer.write(data)
    buffer.flush()


def _arg(argv: list[str], name: str, default: str | None = None) -> str | None:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
        return argv[index + 1]
    return default


def _fail_args(text: str) -> int:
    print(text, file=sys.stderr)
    write_json(error_message("cli", "bad_input", text))
    return exit_code("bad_input")


def _parse(argv: list[str]) -> EnlargeRequest | int:
    if "--json-progress" not in argv:
        print(_USAGE, file=sys.stderr)
        return 2
    source = _arg(argv, "--input")
    if not source:
        return _fail_args("input is required")
    if "--scale" in argv and "--long-side" in argv:
        return _fail_args("pass scale or long-side, not both")
    long_side = 0
    scale = 4
    if "--long-side" in argv:
        raw_side = _arg(argv, "--long-side")
        if raw_side is None:
            return _fail_args("long-side needs a pixel size")
        try:
            long_side = int(raw_side)
        except ValueError:
            return _fail_args("long-side must be an integer")
        if long_side < 1:
            return _fail_args("long-side must be a positive size")
        scale = 0
    elif "--scale" in argv:
        raw_scale = _arg(argv, "--scale")
        if raw_scale is None:
            return _fail_args("scale must be 2 or 4")
        try:
            scale = int(raw_scale)
        except ValueError:
            return _fail_args("scale must be 2 or 4")
        if scale not in (2, 4):
            return _fail_args("scale must be 2 or 4")
    preset = _arg(argv, "--preset", "") or ""
    model = _arg(argv, "--model", "") or ""
    if not preset and not model:
        return _fail_args("preset or model is required")
    tier = _arg(argv, "--quality-tier")
    raw_quality = _arg(argv, "--quality")
    encode = DEFAULT_QUALITY
    if tier is None:
        tier = "fast"
    if raw_quality is None:
        pass
    elif raw_quality in {"fast", "max"}:
        if "--quality-tier" not in argv:
            tier = raw_quality
    else:
        try:
            encode = int(raw_quality)
        except ValueError:
            return _fail_args("quality must be an integer from 1 to 100, or fast or max")
        if encode < 1 or encode > 100:
            return _fail_args("quality must be from 1 to 100")
    if tier not in {"fast", "max"}:
        return _fail_args("quality tier must be fast or max")
    raw_timeout = _arg(argv, "--lock-timeout", "600") or "600"
    try:
        lock_timeout = float(raw_timeout)
    except ValueError:
        return _fail_args("lock-timeout must be a number of seconds")
    if lock_timeout < 0:
        return _fail_args("lock-timeout must be zero or more")
    raw_parent = _arg(argv, "--parent-lock-pid")
    if raw_parent is None:
        raw_parent = os.environ.get("VOXPRINT_GPU_PARENT_PID", "").strip()
    parent = 0
    if raw_parent:
        try:
            parent = int(raw_parent)
        except ValueError:
            return _fail_args("parent-lock-pid must be a process id")
        if parent <= 0:
            return _fail_args("parent-lock-pid must be a process id")
    output = _arg(argv, "--output", "") or ""
    if not output:
        output = _arg(argv, "--out", "") or ""
    return EnlargeRequest(
        id="cli",
        input=source,
        scale=scale,
        preset=preset,
        model=model,
        quality=tier,
        format=_arg(argv, "--format", "") or "",
        output=output,
        output_dir=_arg(argv, "--output-dir", "") or "",
        open_ui=False,
        lossless="--lossless" in argv,
        encode_quality=encode,
        long_side=long_side,
        lock_timeout=lock_timeout,
        parent_lock_pid=parent,
    )


def _request_message(request: EnlargeRequest) -> dict[str, Any]:
    return {
        "op": "enlarge.request",
        "type": "enlarge.request",
        "id": request.id,
        "input": request.input,
        "scale": request.scale,
        "preset": request.preset,
        "quality": request.quality,
        "model": request.model or None,
        "format": request.format or None,
        "output": request.output or None,
        "output_dir": request.output_dir or None,
        "open_ui": False,
        "lossless": request.lossless,
        "encode_quality": request.encode_quality,
        "long_side": request.long_side or None,
        "lock_timeout": request.lock_timeout,
        "parent_lock_pid": request.parent_lock_pid or None,
    }


def _hello() -> dict[str, Any]:
    return {"op": "hello", "type": "hello", "client": "enlarge", "version": APP_VERSION}


def run_capabilities() -> int:
    """Local capabilities, or the warm server's copy when one is listening."""
    remote = exchange([_hello(), {"op": "capabilities", "type": "capabilities"}], {"capabilities", "error"}, timeout=5.0)
    if remote:
        for message in remote:
            if message_kind(message) == "capabilities":
                body = {key: value for key, value in message.items() if key not in {"op", "type"}}
                write_json(body)
                return 0
    write_json(capabilities())
    return 0


def _relay(request: EnlargeRequest, control: JobControl) -> int | None:
    """Forward to a warm server. ``None`` means nothing is listening."""
    if control.is_cancelled():
        write_json(error_message(request.id, "cancelled", "cancelled"))
        return exit_code("cancelled")
    remote = exchange([_hello(), _request_message(request)], {"done", "error"}, timeout=max(30.0, request.lock_timeout + 30.0))
    if remote is None:
        return None
    code = 0
    for message in remote:
        kind = message_kind(message)
        if kind == "hello":
            continue
        write_json(message)
        if kind == "error":
            raw = message.get("code")
            code = exit_code("internal")
            for known in ERROR_CODES:
                if raw == known:
                    code = exit_code(known)
                    break
        elif kind == "done":
            code = 0
    if not any(message_kind(message) in {"done", "error"} for message in remote):
        write_json(error_message(request.id, "internal", "the warm server closed before the job finished"))
        return exit_code("internal")
    if control.is_cancelled() and code == 0:
        return exit_code("cancelled")
    return code


def _install_stop(control: JobControl) -> tuple[Any, Any]:
    def _handle(_signum: int, _frame: object) -> None:
        control.cancel()

    previous_int = signal.signal(signal.SIGINT, _handle)
    previous_term = signal.signal(signal.SIGTERM, _handle)
    return previous_int, previous_term


def run_contract(argv: list[str]) -> int:
    """One headless job. Flags match ``docs/CLI.md``. This function does not construct a window."""
    control = JobControl()
    previous_int, previous_term = _install_stop(control)
    try:
        parsed = _parse(argv)
        if isinstance(parsed, int):
            return parsed
        if control.is_cancelled():
            write_json(error_message(parsed.id, "cancelled", "cancelled"))
            return exit_code("cancelled")
        resolve_exe()
        dry_run = "--dry-run" in argv
        if dry_run:
            return run_job(parsed, write_json, control, dry_run=True)
        forwarded = _relay(parsed, control)
        if forwarded is not None:
            return forwarded
        if control.is_cancelled():
            write_json(error_message(parsed.id, "cancelled", "cancelled"))
            return exit_code("cancelled")
        return run_job(parsed, write_json, control)
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
