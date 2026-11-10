from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import main
from enlarge import suite_runtime
from enlarge.suite_runtime import ModelInstallError, SuiteInstallError, install_model


def test_install_model_downloads_then_prepares(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[object] = []
    monkeypatch.setattr(suite_runtime, "_mdl_download", lambda ids=(): calls.append(("mdl", ids)))
    produced = tmp_path / "model.onnx"

    def fake_prepare(model_id: str, progress: object = None) -> Path:
        calls.append(("prepare", model_id))
        return produced

    monkeypatch.setattr(suite_runtime, "prepare_model", fake_prepare)
    messages: list[str] = []
    assert install_model("4x-nomos2-hq-dat2", messages.append) == produced
    assert calls == [("mdl", ("4x-nomos2-hq-dat2",)), ("prepare", "4x-nomos2-hq-dat2")]
    assert messages


def test_install_model_error_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ModelInstallError) as unknown:
        install_model("no-such-model")
    assert unknown.value.code == "model_missing"

    def fail(ids: tuple[str, ...] = ()) -> None:
        raise SuiteInstallError("voxprint-mdl download --app enlarge failed (3)")

    monkeypatch.setattr(suite_runtime, "_mdl_download", fail)
    assert main._install_model("realesrgan-x4plus-anime6b") == 7

    def bad_prepare(model_id: str, progress: object = None) -> Path:
        raise SuiteInstallError("conversion failed")

    monkeypatch.setattr(suite_runtime, "_mdl_download", lambda ids=(): None)
    monkeypatch.setattr(suite_runtime, "prepare_model", bad_prepare)
    assert main._install_model("realesrgan-x4plus-anime6b") == 1


def test_mdl_download_names_each_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(suite_runtime, "_runtime_python", lambda: "python")
    monkeypatch.setattr(suite_runtime, "_mdl_manifest", lambda: tmp_path / "mdl.json")

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(suite_runtime.subprocess, "run", fake_run)
    suite_runtime._mdl_download(("4x-nomos2-hq-dat2",))
    assert seen[0][-2:] == ["--model", "4x-nomos2-hq-dat2"]
    assert "download" in seen[0]
