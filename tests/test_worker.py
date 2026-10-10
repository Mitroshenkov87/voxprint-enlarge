"""Lanczos stand-in, the worker process, the GPU lock, and the VRAM rule."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from voxprint_engine.gpu_lock import gpu_job, is_stale, lock_path, read_lock, release, try_acquire
from voxprint_engine.thermal import decide
from voxprint_engine.vram import choose_tile, note_spill, parse_nvidia_smi, vram_budget_gb
from voxprint_engine.worker import UpscaleRequest, WorkerClient
from voxprint_engine.worker.lanczos import lanczos_resize


def _picture(path: Path, size: tuple[int, int] = (40, 24)) -> None:
    Image.new("RGB", size, (20, 80, 140)).save(path)


def test_lanczos_scales_and_refuses_video(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    dst = tmp_path / "out.avif"
    _picture(src, (30, 20))
    width, height = lanczos_resize(src, dst, 2, "avif", 60, double=False)
    assert (width, height) == (60, 40)
    assert Image.open(dst).size == (60, 40)
    doubled = tmp_path / "double.jpg"
    width, height = lanczos_resize(src, doubled, 2, "jpeg", 80, double=True)
    assert (width, height) == (120, 80)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="images only"):
        lanczos_resize(video, tmp_path / "no.avif", 2, "avif", 60)


def test_vram_budget_and_tiles() -> None:
    assert vram_budget_gb(10, 24) == pytest.approx(8.0)
    assert vram_budget_gb(10, 40) == pytest.approx(6.8)
    assert vram_budget_gb(1, 8) == 0
    assert choose_tile(0.2, model_tile=256, user_tile=0) == 128
    assert choose_tile(10, model_tile=256, user_tile=0) == 256
    assert choose_tile(1, model_tile=256, user_tile=512) == 512
    assert note_spill(4, spilled=True) == 2
    assert note_spill(1, spilled=True) == 1
    assert note_spill(4, spilled=False) == 4
    parsed = parse_nvidia_smi("24576, 8192\n11264, 1024\n", index=1)
    assert parsed is not None
    assert parsed[0] == pytest.approx(11.0)


def test_gpu_lock_blocks_a_live_owner_and_takes_a_stale_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "voxprint-gpu.lock"
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(target))
    assert try_acquire("preview", eta_s=60) is None
    held = read_lock()
    assert held is not None
    assert held["owner"] == "enlarge"
    assert held["pid"] == os.getpid()
    assert str(held["started"]).endswith("Z")
    # Re-entrant for this process.
    assert try_acquire("again") is None
    release()
    assert read_lock() is None

    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    target.write_text(json.dumps({
        "owner": "movie-dubber",
        "pid": os.getpid(),
        "job": "dub",
        "started": future,
        "eta": future,
    }), encoding="utf-8")
    holder = try_acquire("preview")
    assert holder is not None and holder["owner"] == "movie-dubber"
    with pytest.raises(TimeoutError):
        with gpu_job("preview", timeout=0.05, poll=0.01):
            pass

    old = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    target.write_text(json.dumps({
        "owner": "movie-dubber",
        "pid": os.getpid(),
        "job": "dub",
        "started": old,
        "eta": old,
    }), encoding="utf-8")
    assert is_stale(read_lock() or {})
    assert try_acquire("preview") is None
    release()

    target.write_text(json.dumps({
        "owner": "movie-dubber",
        "pid": 2**31 - 1,
        "job": "gone",
        "started": future,
        "eta": future,
    }), encoding="utf-8")
    assert try_acquire("preview") is None
    release()
    assert lock_path() == target


def test_worker_process_resizes_and_releases_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_FAKE_DELAY", "0.4")
    src = tmp_path / "photo.png"
    dst = tmp_path / "photo_x2.avif"
    _picture(src)
    seen: list[bool] = []

    def watch() -> None:
        for _ in range(30):
            data = read_lock()
            if data and data.get("owner") == "enlarge":
                seen.append(True)
                return
            time.sleep(0.05)

    watcher = threading.Thread(target=watch)
    watcher.start()
    request = UpscaleRequest(
        id="job-1",
        src=str(src),
        dst=str(dst),
        scale=2,
        format="avif",
        quality=60,
        model_id="4x-nomos2-realplksr-dysample",
        tile=0,
        model_tile=256,
        device="cpu",
    )
    with WorkerClient() as client:
        result = client.upscale(request, timeout=20)
    watcher.join(timeout=2)
    assert result.ok
    assert result.engine == "lanczos"
    assert result.model_id == "4x-nomos2-realplksr-dysample"
    assert (result.width, result.height) == (80, 48)
    assert Image.open(dst).size == (80, 48)
    assert result.tile == 128  # cpu budget is 0, so the ladder picks the smallest tile
    assert result.next_batch == 1
    assert seen == [True]
    assert read_lock() is None
    assert not (Path(os.environ["VOXPRINT_HOME"]) / "models" / "4xNomos2_realplksr_dysample.pth").exists()


def test_thermal_rule() -> None:
    early = decide(elapsed_s=60, median_c=90, throttling_s=120, paused=False)
    assert early.action == "run"
    hot = decide(elapsed_s=3 * 60 * 60, median_c=83, throttling_s=0, paused=False)
    assert hot.action == "pause"
    throttle = decide(elapsed_s=3 * 60 * 60, median_c=70, throttling_s=61, paused=False)
    assert throttle.action == "pause"
    cool = decide(elapsed_s=3 * 60 * 60, median_c=75, throttling_s=0, paused=True)
    assert cool.action == "resume"
    waiting = decide(elapsed_s=3 * 60 * 60, median_c=80, throttling_s=0, paused=True)
    assert waiting.action == "pause"
