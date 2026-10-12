"""MDL manifests, the shared-root guard, and the worker parent lock."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from enlarge.suite_runtime import release_runtime
from voxprint_common.gpu_probe import VramTier
from voxprint_common.mdl.catalog import load_manifest, plan_files
from voxprint_engine.worker.client import WorkerClient

ROOT = Path(__file__).resolve().parents[1]


def test_schema1_manifest_plans_the_default_model_and_its_hook(tmp_path: Path) -> None:
    manifest = load_manifest(ROOT / "mdl.json")
    planned = plan_files([manifest], VramTier.GB8, models_root=tmp_path / "models")
    assert [item.model_id for item in planned] == ["4x-nomos2-realplksr-dysample"]
    assert planned[0].size == 29715988
    assert planned[0].destination.name == planned[0].sha256
    hook = planned[0].hooks[0].hook
    assert hook.argv == ("voxprint-enlarge-cli", "--prepare-model", "{id}")
    assert hook.timeout_s == 1800
    assert hook.produces == ("model.onnx",)


def test_app_models_manifest_publishes_post_install() -> None:
    manifest = load_manifest(ROOT / "voxprint-app-models.json")
    assert manifest.app == "enlarge"
    assert manifest.app_hook is not None
    assert manifest.app_hook.argv == ("voxprint-enlarge-cli", "--prepare-model", "{id}")
    assert manifest.app_hook.produces == ("model.onnx",)
    assert manifest.app_models is not None
    assert manifest.app_models.tier_ids("8gb") == ("4x-nomos2-realplksr-dysample",)


def test_release_does_not_delete_a_shared_root_or_another_apps_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    marker = models / "llm" / "gemma.bin"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(models))
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    root = tmp_path / "shared"
    root.mkdir()
    payload = {
        "schema": 1,
        "components": {
            "gemma-4-12b-it-q4": {
                "kind": "model",
                "version": "1",
                "path": str(models),
                "sha256": "ab" * 32,
                "refs": ["enlarge", "audiobook-builder"],
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert release_runtime("enlarge", root=root).ok is True
    assert marker.is_file()
    assert release_runtime("audiobook-builder", root=root).ok is True
    assert marker.is_file()
    assert models.is_dir()


def test_worker_declares_its_parent_lock_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def boom(*_args: object, **kwargs: object) -> None:
        env = kwargs.get("env")
        assert isinstance(env, dict)
        seen["pid"] = str(env.get("VOXPRINT_GPU_PARENT_PID", ""))
        raise OSError("stop")

    monkeypatch.setattr("voxprint_engine.worker.client.subprocess.Popen", boom)
    monkeypatch.delenv("VOXPRINT_GPU_PARENT_PID", raising=False)
    with pytest.raises(OSError, match="stop"):
        WorkerClient().start()
    assert seen["pid"] == str(os.getpid())
