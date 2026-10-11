"""Schema 1 manifest migration through voxprint_common, plus Enlarge's installer."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from enlarge.paths import voxprint_home_source, voxprint_home_warning
from enlarge.suite_runtime import ensure_runtime, read_manifest
from main import main
from voxprint_common.atomic import LockTimeout, exclusive_lock
from voxprint_common.gpu_lock import pid_exists
from voxprint_common.runtime_manifest import load, lock_path


def _write(root: Path, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_canvas_resources_migrate_and_keep_unknown_keys(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    digest = "a" * 64
    _write(
        root,
        {
            "version": 1,
            "note": "keep-me",
            "resources": {
                "ffmpeg-8": {
                    "kind": "tool",
                    "version": "8.0",
                    "path": "runtime/ffmpeg-8",
                    "files": {"ffmpeg": digest},
                    "sha256": "not-the-digest",
                    "installed_at": "2026-01-01T00:00:00+00:00",
                    "refs": ["canvas"],
                    "vendor": "suite",
                }
            },
        },
    )
    manifest = load(root / "manifest.json")
    text = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert text["schema"] == 1
    assert text["note"] == "keep-me"
    record = manifest.components["ffmpeg-8"]
    assert record.refs == ["canvas"]
    assert record.extra["vendor"] == "suite"
    assert record.sha256 == digest


def test_old_enlarge_holders_become_refs(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    _write(
        root,
        {
            "schema": 1,
            "kept": True,
            "components": {
                "python-3.14": {"holders": ["enlarge"], "version": "3.14", "kind": "runtime"},
                "model": {"holders": ["enlarge"], "version": "photo"},
            },
        },
    )
    manifest = load(root / "manifest.json")
    assert manifest.extra["kept"] is True
    python = manifest.components["python-3.14"]
    assert python.kind == "runtime"
    assert python.refs == ["enlarge"]
    assert "holders" not in python.to_dict()
    assert manifest.components["model"].kind == "model"
    assert manifest.components["model"].refs == ["enlarge"]


def test_lock_times_out_while_another_holder_is_alive(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    (root / "manifest.json").write_text('{"schema": 1, "components": {}}\n', encoding="utf-8")
    with exclusive_lock(lock_path(root / "manifest.json")):
        with pytest.raises(LockTimeout):
            load(root / "manifest.json", timeout=0.2)


def test_a_live_pid_is_detected_without_signalling_it() -> None:
    assert pid_exists(os.getpid()) is True
    assert pid_exists(0) is False
    assert pid_exists(-1) is False


def test_ensure_does_not_overwrite_a_component_another_app_refs(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    _write(
        root,
        {
            "schema": 1,
            "components": {
                "python-3.14": {
                    "kind": "runtime",
                    "version": "3.14",
                    "path": "python",
                    "sha256": "someone-elses-build",
                    "installed_at": "2026-01-01T00:00:00+00:00",
                    "refs": ["canvas"],
                }
            },
        },
    )

    def probe(_path: Path) -> bool:
        return True

    def install(path: Path, progress) -> None:
        progress("install")
        (path / "python").write_text("ready", encoding="utf-8")

    probes = {"python": probe, "gpu": probe, "model": probe}
    installers = {"python": install, "gpu": install, "model": install}
    report = ensure_runtime("enlarge", root=root, allow_skip=False, probes=probes, installers=installers)
    assert report.ok is True
    manifest = read_manifest(root)
    components = manifest["components"]
    assert isinstance(components, dict)
    original = components["python-3.14"]
    assert isinstance(original, dict)
    assert original["sha256"] == "someone-elses-build"
    assert original["refs"] == ["canvas"]
    side = [key for key in components if str(key).startswith("python-3.14-")]
    assert side
    side_row = components[side[0]]
    assert isinstance(side_row, dict)
    assert "enlarge" in side_row["refs"]


def test_stray_home_warns_only_when_the_default_manifest_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    default = tmp_path / "default"
    custom = tmp_path / "custom"
    monkeypatch.setattr("enlarge.paths.default_voxprint_home", lambda: default)
    monkeypatch.setenv("VOXPRINT_HOME", str(custom))
    assert voxprint_home_source() == "env"
    assert voxprint_home_warning() is None
    (default / "shared").mkdir(parents=True)
    (default / "shared" / "manifest.json").write_text("{}\n", encoding="utf-8")
    warning = voxprint_home_warning()
    assert warning is not None
    assert str(custom) in warning
    assert str(default) in warning
    assert main(["--capabilities", "--json"]) == 0
    captured = capsys.readouterr()
    assert warning in captured.err
    payload = json.loads(captured.out)
    assert payload["voxprint_home"] == str(custom)
    assert payload["voxprint_home_source"] == "env"
    monkeypatch.setenv("VOXPRINT_HOME", str(default))
    assert voxprint_home_warning() is None
