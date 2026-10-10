"""Queue resume, model refcount, TensorRT cache wording, and cancel."""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from enlarge.batch import BatchControl, run_batch
from enlarge.credits import credit_sections
from enlarge.queue_state import dump_queue, load_queue, still_open
from voxprint_engine.registry import load_builtin
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.session import trt_build_message
from voxprint_engine.store import StoreError, delete_weight, register, unregister, weight_report
from voxprint_engine.worker import UpscaleRequest


def _model(**updates: object) -> ModelManifest:
    base = dict(
        id="demo",
        name="Demo",
        arch="test",
        scale=4,
        role="upscale",
        licence="MIT",
        author="A",
        attribution="A",
        source_url="https://example.com/model",
        sha256="ab" * 32,
        vram_mb=1,
        tile=0,
        filename="demo.bin",
        summary="",
        selectable=True,
        download_url="https://example.com/demo.bin",
    )
    base.update(updates)
    return ModelManifest(**base)  # type: ignore[arg-type]


def test_queue_marks_a_crash_and_skips_finished_files(tmp_path: Path) -> None:
    done = tmp_path / "done.avif"
    done.write_bytes(b"ok")
    missing = tmp_path / "missing.avif"
    path = tmp_path / "queue.json"
    dump_queue(
        [
            {"id": "a", "src": "a.png", "dst": done, "status": "Done", "message": "ok"},
            {"id": "b", "src": "b.png", "dst": missing, "status": "Running", "message": ""},
        ],
        path,
    )
    rows = load_queue(path)
    assert rows[0]["status"] == "Done"
    assert rows[1]["status"] == "Interrupted"
    assert still_open(rows[0]) is False
    assert still_open(rows[1]) is True


def test_credits_follow_the_registry() -> None:
    sections = dict(credit_sections(load_builtin().models))
    assert "CC-BY-4.0" in sections
    assert "BSD-3-Clause" in sections
    assert any(model.id == "realesrgan-x4plus-anime6b" for model in sections["BSD-3-Clause"])


def test_delete_respects_the_users_refcount(tmp_path: Path) -> None:
    payload = b"weight"
    model = _model(sha256=hashlib.sha256(payload).hexdigest())
    target = tmp_path / model.filename
    target.write_bytes(payload)
    assert weight_report(model, tmp_path) == ("ready", "sha256 matches")
    target.write_bytes(b"tampered")
    assert weight_report(model, tmp_path)[0] == "mismatch"
    target.write_bytes(payload)
    register("enlarge", tmp_path)
    register("clarity", tmp_path)
    with pytest.raises(StoreError, match="clarity"):
        delete_weight(model, tmp_path)
    assert target.is_file()
    unregister("clarity", tmp_path)
    delete_weight(model, tmp_path)
    assert not target.is_file()


def test_trt_message_names_a_cold_and_a_warm_cache(tmp_path: Path) -> None:
    assert trt_build_message(tmp_path / "missing") == "building the TensorRT engine"
    warm = tmp_path / "warm"
    warm.mkdir()
    (warm / "model.engine").write_bytes(b"engine")
    assert trt_build_message(warm) == "loading the cached TensorRT engine"


def test_cancel_stops_the_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_FAKE_DELAY", "2")
    source = tmp_path / "slow.png"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(source)
    request = UpscaleRequest(
        id="slow",
        src=str(source),
        dst=str(tmp_path / "out.webp"),
        scale=2,
        format="webp",
        quality=60,
        model_id="4x-nomos2-realplksr-dysample",
    )
    control = BatchControl()
    seen: list[str] = []

    def _later() -> None:
        time.sleep(0.2)
        control.cancel()

    threading.Thread(target=_later, daemon=True).start()
    started = time.monotonic()
    run_batch([request], lambda *_parts: None, lambda *_parts: seen.append("item"), control.is_cancelled, control)
    assert time.monotonic() - started < 1.8
    assert seen == ["item"]
    assert not (tmp_path / "out.webp").exists()
