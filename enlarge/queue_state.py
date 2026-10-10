"""The Run queue on disk, so a crash can continue from the last finished image."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enlarge.paths import app_home

FINISHED = "Done"


def queue_path() -> Path:
    return app_home() / "queue.json"


def dump_queue(rows: list[dict[str, object]], path: Path | None = None) -> None:
    target = path or queue_path()
    items = [_item(row) for row in rows]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"items": items}, indent=1), encoding="utf-8")


def load_queue(path: Path | None = None) -> list[dict[str, Any]]:
    """Rows from the last session. A ``Running`` row becomes ``Interrupted``."""
    target = path or queue_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    raw = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src") or "")
        dst = str(item.get("dst") or "")
        if not src or not dst:
            continue
        status = str(item.get("status") or "Waiting")
        message = str(item.get("message") or "")
        if status == "Running":
            status = "Interrupted"
            message = message or "stopped before this image finished"
        rows.append({
            "id": str(item.get("id") or ""),
            "src": src,
            "dst": dst,
            "status": status,
            "message": message,
        })
    return rows


def still_open(row: dict[str, object]) -> bool:
    """True when Run should send this image again."""
    if str(row.get("status") or "") != FINISHED:
        return True
    return not Path(str(row.get("dst") or "")).is_file()


def _item(row: dict[str, object]) -> dict[str, str]:
    return {
        "id": str(row.get("id") or ""),
        "src": str(row.get("src") or ""),
        "dst": str(row.get("dst") or ""),
        "status": str(row.get("status") or "Waiting"),
        "message": str(row.get("message") or ""),
    }
