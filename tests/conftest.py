"""Offscreen Qt and private Voxprint folders for every test."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "voxprint"))
    monkeypatch.setenv("VOXPRINT_ENLARGE_HOME", str(tmp_path / "enlarge"))
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "voxprint-gpu.lock"))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    monkeypatch.delenv("VOXPRINT_FAKE_DELAY", raising=False)
