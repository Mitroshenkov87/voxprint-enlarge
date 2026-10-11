"""``QLocalServer`` for protocol 1. ``--serve --hidden`` keeps the window closed until ``open_ui``."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from enlarge.ipc.capabilities import capabilities
from enlarge.ipc.jobs import JobControl, run_job
from enlarge.ipc.naming import listen_name, prepare_endpoint, tighten_socket
from enlarge.ipc.protocol import (
    EnlargeRequest,
    dumps,
    error_message,
    hello_message,
    loads,
    message_kind,
    request_from_message,
)
from enlarge.appinfo import APP_DISPLAY_NAME, product_version
from enlarge.ui.window import MainWindow
from voxprint_engine.nvidia_smi import assess
from voxprint_engine.providers import ci_lanczos


class _Client:
    def __init__(self, socket: QLocalSocket) -> None:
        self.socket = socket
        self.buffer = b""


class IpcServer(QObject):
    """One listener. Jobs run one at a time and share the suite GPU lock."""

    _kick = Signal()

    def __init__(self) -> None:
        """Build the listener. The window stays None until open_ui."""
        super().__init__()
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self.server.newConnection.connect(self._accept)
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._flush)
        self._kick.connect(self._start_next, Qt.ConnectionType.QueuedConnection)
        self.outbox: queue.Queue[tuple[_Client, dict[str, Any]]] = queue.Queue()
        self.clients: list[_Client] = []
        self.pending: list[tuple[_Client, EnlargeRequest, int]] = []
        self.controls: dict[str, JobControl] = {}
        self.owners: dict[str, _Client] = {}
        self.active = False
        self.window: MainWindow | None = None
        self.name = ""
        self.error = ""

    def listen(self) -> bool:
        """Bind the per-user socket or pipe. False means the endpoint is taken."""
        self.name = prepare_endpoint()
        if not self.server.listen(self.name):
            self.error = self.server.errorString()
            return False
        tighten_socket(self.name)
        self.timer.start()
        return True

    def endpoint(self) -> str:
        """The listen name passed to QLocalServer."""
        return self.name or listen_name()

    def close(self) -> None:
        """Stop listening and drop the current client."""
        self.timer.stop()
        for control in self.controls.values():
            control.cancel()
        for client in list(self.clients):
            client.socket.disconnectFromServer()
        self.server.close()

    def _accept(self) -> None:
        while True:
            socket = self.server.nextPendingConnection()
            if socket is None:
                return
            client = _Client(socket)
            self.clients.append(client)
            socket.readyRead.connect(lambda client=client: self._read(client))
            socket.disconnected.connect(lambda client=client: self._drop(client))

    def _drop(self, client: _Client) -> None:
        if client in self.clients:
            self.clients.remove(client)
        kept: list[tuple[_Client, EnlargeRequest, int]] = []
        for item in self.pending:
            if item[0] is client:
                self._send(client, error_message(item[1].id, "cancelled", "cancelled"))
            else:
                kept.append(item)
        self.pending = kept
        for job_id, owner in list(self.owners.items()):
            if owner is client:
                control = self.controls.get(job_id)
                if control is not None:
                    control.cancel()

    def _read(self, client: _Client) -> None:
        client.buffer += bytes(client.socket.readAll().data())
        while b"\n" in client.buffer:
            raw, _, client.buffer = client.buffer.partition(b"\n")
            self._on_line(client, raw)

    def _on_line(self, client: _Client, raw: bytes) -> None:
        try:
            message = loads(raw)
        except (UnicodeError, ValueError) as exc:
            self._send(client, error_message("", "internal", str(exc)))
            return
        kind = message_kind(message)
        if kind == "hello":
            self._send(client, hello_message(product_version()))
            return
        if kind == "capabilities":
            payload = capabilities()
            payload["op"] = "capabilities"
            payload["type"] = "capabilities"
            self._send(client, payload)
            return
        if kind == "cancel":
            job_id = message.get("id")
            control = self.controls.get(job_id) if isinstance(job_id, str) else None
            if control is not None:
                control.cancel()
            return
        if kind == "enlarge.request":
            self._enqueue(client, message)
            return
        job_id = message.get("id")
        self._send(client, error_message(job_id if isinstance(job_id, str) else "", "internal", "unknown op"))

    def _enqueue(self, client: _Client, message: dict[str, Any]) -> None:
        try:
            request = request_from_message(message)
        except ValueError as exc:
            job_id = message.get("id")
            self._send(client, error_message(job_id if isinstance(job_id, str) else "", "bad_input", str(exc)))
            return
        if request.open_ui and (ci_lanczos() or not assess(APP_DISPLAY_NAME).message):
            self._open_ui(request.input)
        position = (1 if self.active else 0) + len(self.pending)
        self.pending.append((client, request, position))
        self._start_next()

    def _open_ui(self, path: str) -> None:
        if self.window is None:
            self.window = MainWindow()
        self.window.add_paths([Path(path)])
        self.window.show()

    def _start_next(self) -> None:
        if self.active or not self.pending:
            return
        client, request, position = self.pending.pop(0)
        control = JobControl()
        self.controls[request.id] = control
        self.owners[request.id] = client
        self.active = True

        def _run() -> None:
            try:
                run_job(request, lambda message: self.outbox.put((client, message)), control, queue_pos=position)
            finally:
                self.controls.pop(request.id, None)
                self.owners.pop(request.id, None)
                self.active = False
                self._kick.emit()

        threading.Thread(target=_run, name=f"enlarge-ipc-{request.id}", daemon=True).start()

    def _send(self, client: _Client, message: dict[str, Any]) -> None:
        self.outbox.put((client, message))
        self._flush()

    def _flush(self) -> None:
        while True:
            try:
                client, message = self.outbox.get_nowait()
            except queue.Empty:
                return
            socket = client.socket
            if socket.state() != QLocalSocket.LocalSocketState.ConnectedState:
                continue
            socket.write(dumps(message))
            socket.flush()
