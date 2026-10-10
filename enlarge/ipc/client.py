"""Blocking protocol-1 client used when a warm ``--serve`` process is already listening.

The socket is the stdlib. This module does not import PySide6, so
``--capabilities`` and ``--dry-run`` do not need libEGL.
"""
from __future__ import annotations

import ctypes
import socket
import sys
import time
from ctypes import wintypes
from typing import Any, Protocol

if sys.platform == "win32":
    import msvcrt
else:
    msvcrt = None

from enlarge.ipc.naming import listen_name
from enlarge.ipc.protocol import dumps, loads, message_kind

_CONNECT_S = 0.2


class _Link(Protocol):
    def write(self, data: bytes) -> None:
        """Send bytes to the warm server."""

    def read_some(self, timeout: float) -> bytes | None:
        """Return a chunk, ``b""`` on timeout, or ``None`` when the peer is gone."""

    def close(self) -> None:
        """Release the connection."""


class _UnixLink:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def write(self, data: bytes) -> None:
        self._sock.sendall(data)

    def read_some(self, timeout: float) -> bytes | None:
        self._sock.settimeout(timeout)
        try:
            chunk = self._sock.recv(65536)
        except TimeoutError:
            return b""
        except OSError:
            return None
        if not chunk:
            return None
        return chunk

    def close(self) -> None:
        self._sock.close()


class _PipeLink:
    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def write(self, data: bytes) -> None:
        self._handle.write(data)
        self._handle.flush()

    def read_some(self, timeout: float) -> bytes | None:
        if msvcrt is None:
            return None
        windll = getattr(ctypes, "WinDLL")
        kernel32 = windll("kernel32", use_last_error=True)
        raw = msvcrt.get_osfhandle(self._handle.fileno())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            avail = wintypes.DWORD(0)
            ok = kernel32.PeekNamedPipe(
                wintypes.HANDLE(raw),
                None,
                0,
                None,
                ctypes.byref(avail),
                None,
            )
            if not ok:
                return None
            if avail.value:
                try:
                    return self._handle.read(int(avail.value)) or None
                except OSError:
                    return None
            time.sleep(0.02)
        return b""

    def close(self) -> None:
        self._handle.close()


def _connect() -> _Link | None:
    name = listen_name()
    if sys.platform == "win32":
        path = "\\\\.\\pipe\\" + name
        try:
            handle = open(path, "r+b", buffering=0)
        except OSError:
            return None
        return _PipeLink(handle)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(_CONNECT_S)
    try:
        sock.connect(name)
    except OSError:
        sock.close()
        return None
    return _UnixLink(sock)


def _normalize(message: dict[str, Any]) -> dict[str, Any]:
    """Published servers send ``op``. The CLI contract also carries ``type``."""
    body = dict(message)
    kind = message_kind(body)
    if kind:
        body.setdefault("op", kind)
        body.setdefault("type", kind)
    if kind == "progress" and "eta_s" not in body:
        body["eta_s"] = None
    return body


def exchange(
    outgoing: list[dict[str, Any]],
    until: set[str],
    timeout: float = 120.0,
) -> list[dict[str, Any]] | None:
    """Send ``outgoing`` and read until a message whose kind is in ``until``.

    ``None`` means nothing is listening for this user.

    Args:
        outgoing: JSON objects, in order, each written as one line.
        until: Stop when a message kind is in this set.
        timeout: Seconds to wait after the connection succeeds.

    Returns:
        The messages read, or ``None`` when connect fails.
    """
    link = _connect()
    if link is None:
        return None
    try:
        for message in outgoing:
            link.write(dumps(message))
        buffer = b""
        messages: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chunk = link.read_some(min(0.05, max(0.0, deadline - time.monotonic())))
            if chunk is None:
                break
            if chunk:
                buffer += chunk
            while b"\n" in buffer:
                raw, _, buffer = buffer.partition(b"\n")
                if not raw.strip():
                    continue
                try:
                    parsed = loads(raw)
                except ValueError:
                    continue
                message = _normalize(parsed)
                messages.append(message)
                if message_kind(message) in until:
                    return messages
        return messages
    finally:
        link.close()
