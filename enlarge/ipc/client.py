"""Blocking protocol-1 client used when a warm ``--serve`` process is already listening."""
from __future__ import annotations

import time
from typing import Any

from PySide6.QtNetwork import QLocalSocket
from PySide6.QtWidgets import QApplication

from enlarge.ipc.naming import listen_name
from enlarge.ipc.protocol import dumps, loads, message_kind

_CONNECT_MS = 200


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


def exchange(outgoing: list[dict[str, Any]], until: set[str], timeout: float = 120.0) -> list[dict[str, Any]] | None:
    """Send ``outgoing`` and read until a message whose kind is in ``until``.

    ``None`` means nothing is listening for this user. The socket is not created.
    """
    running = QApplication.instance()
    app = running if isinstance(running, QApplication) else QApplication([])
    sock = QLocalSocket()
    sock.connectToServer(listen_name())
    if not sock.waitForConnected(_CONNECT_MS):
        sock.abort()
        return None
    try:
        for message in outgoing:
            sock.write(dumps(message))
        sock.flush()
        sock.waitForBytesWritten(2000)
        buffer = b""
        messages: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if sock.waitForReadyRead(30):
                buffer += bytes(sock.readAll().data())
            elif sock.bytesAvailable() > 0:
                buffer += bytes(sock.readAll().data())
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
            state = sock.state()
            if state == QLocalSocket.LocalSocketState.UnconnectedState and not buffer:
                break
        return messages
    finally:
        sock.disconnectFromServer()
        sock.abort()
