"""Suite GPU lock: ``%TEMP%/voxprint-gpu.lock``.

JSON object ``{owner, pid, job, started, eta}`` with ISO-8601 UTC timestamps.
The file is stale when its pid is gone or its ``eta`` is more than two hours
in the past. Enlarge's owner name is ``enlarge``. A job this process already
holds is re-entered and released only by the outermost block.
``VOXPRINT_GPU_PARENT_PID``, when it matches the lock file's pid, runs inside
that lock and does not release it, whatever the owner name is.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import psutil

OWNER = "enlarge"
LOCK_NAME = "voxprint-gpu.lock"
STALE_AFTER = timedelta(hours=2)

_depth = 0


class GpuLockTimeout(TimeoutError):
    """Another Voxprint program still holds the GPU and the wait expired."""


def lock_path() -> Path:
    """Path of the suite GPU lock file."""
    override = os.environ.get("VOXPRINT_GPU_LOCK")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / LOCK_NAME


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, (int, str)):
        return False
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    return bool(psutil.pid_exists(number))


def read_lock(path: Path | None = None) -> dict[str, Any] | None:
    """The lock object, ``{}`` when the file is unreadable, or None when there is no file."""
    target = path or lock_path()
    try:
        text = target.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def is_stale(data: dict[str, Any], now: datetime | None = None) -> bool:
    """Stale when the pid is gone, or ``eta`` (else ``started``) is more than two hours ago."""
    if not data:
        return True
    if not _pid_alive(data.get("pid")):
        return True
    moment = now or _now()
    eta = _parse(data.get("eta")) or _parse(data.get("started"))
    if eta is None:
        return True
    return moment - eta > STALE_AFTER


def _ours(data: dict[str, Any] | None, owner: str | None = None) -> bool:
    if not data:
        return False
    expected = OWNER if owner is None else owner
    return data.get("owner") == expected and data.get("pid") == os.getpid()


def _declared_parent_pid() -> int | None:
    """``VOXPRINT_GPU_PARENT_PID`` is a caller that already holds the suite lock.

    The pid is checked against the lock file. The owner name is not.
    """
    raw = os.environ.get("VOXPRINT_GPU_PARENT_PID", "").strip()
    if not raw:
        return None
    try:
        number = int(raw)
    except ValueError:
        return None
    if number <= 0:
        return None
    return number


def _pid_value(pid: object) -> int | None:
    if isinstance(pid, bool) or not isinstance(pid, (int, str)):
        return None
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def parent_lock_matches(pid: int, path: Path | None = None) -> bool:
    """True when the suite lock file exists and its ``pid`` is ``pid``."""
    if pid <= 0:
        return False
    data = read_lock(path)
    if not data:
        return False
    return _pid_value(data.get("pid")) == pid


def _held_by_declared_parent(data: dict[str, Any] | None) -> bool:
    """A caller passed the pid of the process that already holds the lock.

    Enlarge runs inside that lock and does not release it, whoever the owner is.
    """
    declared = _declared_parent_pid()
    if declared is None or not data:
        return False
    return _pid_value(data.get("pid")) == declared


def _held_by_parent(data: dict[str, Any] | None, owner: str | None = None) -> bool:
    """The worker subprocess shares a lock its parent already holds for this job.

    The parent and the child name the same owner. An ``enlarge`` parent lock
    is not a ``clarity`` lock, and the child does not release the parent's file.
    """
    expected = OWNER if owner is None else owner
    if not data or data.get("owner") != expected:
        return False
    pid = data.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, (int, str)):
        return False
    try:
        return int(pid) == os.getppid()
    except (TypeError, ValueError):
        return False


def try_acquire(
    job: str,
    eta_s: float = 600.0,
    path: Path | None = None,
    owner: str | None = None,
) -> dict[str, Any] | None:
    """Create the lock. None means this process holds it; otherwise the holder's data is returned.

    ``owner`` defaults to ``enlarge`` so existing callers stay the same.
    One process has one owner: re-entrancy is the process-global depth.
    A lock held by this process's parent, with the same owner, is shared.
    """
    name = OWNER if owner is None else owner
    target = path or lock_path()
    current = read_lock(target)
    if current is not None:
        if _ours(current, name) or _held_by_parent(current, name) or _held_by_declared_parent(current):
            return None
        if not is_stale(current):
            return current or {"owner": "?"}
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return current or {"owner": "?"}
    moment = _now()
    payload = {
        "owner": name,
        "pid": os.getpid(),
        "job": job,
        "started": _iso(moment),
        "eta": _iso(moment + timedelta(seconds=max(60.0, float(eta_s)))),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return read_lock(target) or {"owner": "?"}
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return None


def release(path: Path | None = None, owner: str | None = None) -> None:
    """Delete the lock when this process owns it."""
    target = path or lock_path()
    current = read_lock(target)
    if _held_by_declared_parent(current):
        return
    if not _ours(current, owner):
        return
    try:
        target.unlink()
    except OSError:
        return


@contextmanager
def gpu_job(
    job: str,
    eta_s: float = 600.0,
    timeout: float | None = None,
    poll: float = 0.25,
    path: Path | None = None,
    cancel: Callable[[], bool] | None = None,
    on_wait: Callable[[dict[str, Any], float], None] | None = None,
    owner: str | None = None,
) -> Iterator[None]:
    """Hold the suite lock for one job. Re-entrant for nested blocks in this process.

    ``owner`` defaults to ``enlarge``. Clarity passes ``owner="clarity"``.
    """
    global _depth
    if _depth > 0:
        _depth += 1
        try:
            yield
        finally:
            _depth -= 1
        return
    started = time.monotonic()
    while True:
        holder = try_acquire(job, eta_s, path, owner)
        if holder is None:
            break
        waited = time.monotonic() - started
        if cancel is not None and cancel():
            raise GpuLockTimeout("cancelled while waiting for the GPU")
        if timeout is not None and waited > timeout:
            who = holder.get("owner", "?")
            what = holder.get("job", "?")
            raise GpuLockTimeout(f"the GPU is busy: {who} / {what}")
        if on_wait is not None:
            on_wait(holder, waited)
        time.sleep(poll)
    _depth = 1
    try:
        yield
    finally:
        _depth = 0
        release(path, owner)
