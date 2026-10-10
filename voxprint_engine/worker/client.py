"""Parent-side client. Starts ``python -m voxprint_engine.worker`` and speaks JSON lines."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from voxprint_engine.worker.protocol import UpscaleRequest, UpscaleResult

_ROOT = Path(__file__).resolve().parents[2]


class WorkerError(RuntimeError):
    """The worker process failed, timed out, or returned an error result."""


class WorkerClient:
    """One worker process. Use it as a context manager so the child is shut down."""

    def __init__(self, python: str | None = None) -> None:
        self.python = python or sys.executable
        self.proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: list[str] = []
        self._reader: threading.Thread | None = None

    def __enter__(self) -> WorkerClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self, timeout: float = 15.0) -> None:
        if self.proc is not None:
            return
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        previous = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + previous if previous else "")
        command = [self.python, "-m", "voxprint_engine.worker"]
        if os.name == "nt":
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(_ROOT),
                env=env,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        else:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(_ROOT),
                env=env,
            )
        self._reader = threading.Thread(target=self._pump, name="enlarge-worker-stdout", daemon=True)
        self._reader.start()
        threading.Thread(target=self._pump_err, name="enlarge-worker-stderr", daemon=True).start()
        ready = self._read(timeout)
        if ready.get("op") != "ready":
            self.close()
            raise WorkerError(f"worker did not become ready: {ready}")

    def upscale(
        self,
        request: UpscaleRequest,
        on_progress: Callable[[float, str], None] | None = None,
        timeout: float = 120.0,
    ) -> UpscaleResult:
        self._send(request.to_message())
        while True:
            message = self._read(timeout)
            op = message.get("op")
            if op == "progress":
                if on_progress is not None:
                    raw_fraction = message.get("fraction")
                    fraction = float(raw_fraction) if isinstance(raw_fraction, (int, float)) and not isinstance(raw_fraction, bool) else 0.0
                    on_progress(fraction, str(message.get("message") or ""))
                continue
            if op in ("result", "error"):
                result = UpscaleResult.from_message(message)
                if not result.ok:
                    raise WorkerError(result.message or "worker error")
                return result
            raise WorkerError(f"unexpected worker message: {message}")

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        if proc.poll() is None and proc.stdin is not None:
            try:
                proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                proc.stdin.flush()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def stderr_text(self) -> str:
        return "".join(self._stderr)[-4000:]

    def _send(self, message: dict[str, object]) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            raise WorkerError("worker is not running")
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def _read(self, timeout: float) -> dict[str, object]:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as exc:
            detail = self.stderr_text()
            raise WorkerError("worker timed out" + (f": {detail}" if detail else "")) from exc
        if line is None:
            detail = self.stderr_text()
            raise WorkerError("worker closed the pipe" + (f": {detail}" if detail else ""))
        try:
            message = json.loads(line)
        except ValueError as exc:
            raise WorkerError(f"worker sent malformed json: {line[:200]}") from exc
        if not isinstance(message, dict):
            raise WorkerError("worker sent a non-object")
        return message

    def _pump(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            self._lines.put(None)
            return
        for line in proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_err(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr.append(line)
