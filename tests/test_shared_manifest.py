"""Schema 1 shared manifest: migration, lock, refs, and side-by-side installs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from voxprint_engine.registry import DEFAULT_MODEL_ID, load_builtin
from voxprint_engine.shared_manifest import (
    SharedManifestError,
    add_ref,
    find_reusable,
    load,
    lock_hash,
    remove_ref,
    reserve,
    save,
    update,
)
from main import main
from voxprint_engine.paths import voxprint_home_source, voxprint_home_warning
from voxprint_engine.suite_runtime import ensure_runtime, read_manifest


def _write(root: Path, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_canvas_resources_migrate_and_keep_unknown_keys(tmp_path: Path) -> None:
    root = tmp_path / "shared"
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
                    "sha256": "abc",
                    "installed_at": "2026-01-01T00:00:00+00:00",
                    "refs": ["canvas"],
                    "vendor": "suite",
                }
            },
        },
    )
    manifest = load(root)
    text = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert text["schema"] == 1
    assert "resources" not in text
    assert text["note"] == "keep-me"
    assert text["version"] == 1
    record = manifest.components["ffmpeg-8"]
    assert record["refs"] == ["canvas"]
    assert record["vendor"] == "suite"
    assert record["sha256"] == "abc"
    with update(root) as fresh:
        add_ref(fresh, "ffmpeg-8", "enlarge")
    again = load(root)
    assert again.extra["note"] == "keep-me"
    assert again.components["ffmpeg-8"]["vendor"] == "suite"
    assert again.components["ffmpeg-8"]["refs"] == ["canvas", "enlarge"]


def test_old_enlarge_holders_become_schema_1_components(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    model = load_builtin(release=True).get(DEFAULT_MODEL_ID)
    assert model is not None
    _write(
        root,
        {
            "schema": 1,
            "kept": True,
            "components": {
                "python": {"holders": ["enlarge"], "version": "3.14"},
                "gpu": {"holders": ["canvas"], "version": "cu130"},
                "model": {"holders": ["enlarge"], "version": DEFAULT_MODEL_ID},
            },
        },
    )
    manifest = load(root)
    assert manifest.extra["kept"] is True
    python = manifest.components["python-3.14"]
    assert python["kind"] == "runtime"
    assert python["refs"] == ["enlarge"]
    assert python["sha256"] == lock_hash("python==3.14")
    assert "holders" not in python
    assert manifest.components["torch-cu130"]["refs"] == ["canvas"]
    assert manifest.components["onnxruntime-gpu-1.31"]["kind"] == "wheels"
    assert manifest.components["tensorrt-10"]["version"] == "10.14.1.48"
    model_id = f"model-{model.sha256[:12]}"
    assert manifest.components[model_id]["refs"] == ["enlarge"]
    assert manifest.components[model_id]["sha256"] == model.sha256.lower()


def test_lock_times_out_while_another_holder_is_alive(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    lock = root / "manifest.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    os.write(fd, f"{os.getpid()}\n".encode("ascii"))
    try:
        with pytest.raises(SharedManifestError, match="timed out"):
            with update(root, timeout=0.2):
                pass
    finally:
        os.close(fd)
        lock.unlink()


def test_failed_rename_keeps_the_previous_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "shared"
    with update(root) as manifest:
        reserve(
            manifest,
            component_id="python-3.14",
            kind="runtime",
            version="3.14",
            path="python",
            sha256="abc",
            app="enlarge",
        )
    original = (root / "manifest.json").read_text(encoding="utf-8")

    def boom(_src: str | Path, _dst: str | Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        save(load(root), root)
    assert (root / "manifest.json").read_text(encoding="utf-8") == original
    tmp = root / "manifest.json.tmp"
    assert tmp.is_file()
    assert "disk full" not in original
    assert json.loads(tmp.read_text(encoding="utf-8"))["schema"] == 1


def test_refs_lifecycle_reports_when_the_component_is_deletable(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    with update(root) as manifest:
        reserve(
            manifest,
            component_id="tensorrt-10",
            kind="wheels",
            version="10.14.1.48",
            path="python",
            sha256="abc",
            app="enlarge",
        )
        add_ref(manifest, "tensorrt-10", "enlarge")
        add_ref(manifest, "tensorrt-10", "canvas")
        assert manifest.components["tensorrt-10"]["refs"] == ["enlarge", "canvas"]
        assert remove_ref(manifest, "tensorrt-10", "enlarge") is False
        assert remove_ref(manifest, "tensorrt-10", "canvas") is True
        assert manifest.components["tensorrt-10"]["refs"] == []


def test_reuse_matches_the_triple_and_a_foreign_sha_installs_beside_it(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    with update(root) as manifest:
        created, reused = reserve(
            manifest,
            component_id="python-3.14",
            kind="runtime",
            version="3.14",
            path="python",
            sha256="abc",
            app="canvas",
        )
        assert reused is False
        assert created["path"] == "python"
    with update(root) as manifest:
        found = find_reusable(manifest, "python-3.14", "3.14", "abc")
        assert found is not None
        again, reused = reserve(
            manifest,
            component_id="python-3.14",
            kind="runtime",
            version="3.14",
            path="python",
            sha256="abc",
            app="enlarge",
        )
        assert reused is True
        assert again["refs"] == ["canvas", "enlarge"]
        side, reused = reserve(
            manifest,
            component_id="python-3.14",
            kind="runtime",
            version="3.14",
            path="python-defdefdefdef",
            sha256="defdefdefdef",
            app="enlarge",
        )
        assert reused is False
        assert manifest.components["python-3.14"]["sha256"] == "abc"
        assert manifest.components["python-3.14"]["refs"] == ["canvas"]
        assert side["path"] == "python-defdefdefdef"
        assert "python-3.14-defdefdefdef" in manifest.components
    with update(root) as manifest:
        remove_ref(manifest, "python-3.14-defdefdefdef", "enlarge")
        replaced, reused = reserve(
            manifest,
            component_id="onnxruntime-gpu-1.31",
            kind="wheels",
            version="1.31.0",
            path="python",
            sha256="one",
            app="enlarge",
        )
        assert reused is False
        replaced, reused = reserve(
            manifest,
            component_id="onnxruntime-gpu-1.31",
            kind="wheels",
            version="1.31.0",
            path="python",
            sha256="two",
            app="enlarge",
        )
        assert reused is False
        assert "onnxruntime-gpu-1.31-two" not in manifest.components
        assert replaced["sha256"] == "two"
        assert manifest.components["onnxruntime-gpu-1.31"]["refs"] == ["enlarge"]


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
    original = manifest["components"]["python-3.14"]
    assert original["sha256"] == "someone-elses-build"
    assert original["refs"] == ["canvas"]
    side = [key for key in manifest["components"] if key.startswith("python-3.14-")]
    assert side
    assert "enlarge" in manifest["components"][side[0]]["refs"]


def test_stray_home_warns_only_when_the_default_manifest_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    default = tmp_path / "default"
    custom = tmp_path / "custom"
    monkeypatch.setattr("voxprint_engine.paths.default_voxprint_home", lambda: default)
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
