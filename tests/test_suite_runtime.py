"""Shared suite runtime: refcount, repair, and the installer commands."""
from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from main import main
from voxprint_common.paths import shared_dir
from voxprint_engine.registry import DEFAULT_MODEL_ID, load_builtin
from voxprint_engine.store import register
from enlarge.suite_runtime import (
    PYTHON_VERSION,
    ComponentStatus,
    RuntimeReport,
    SuiteInstallError,
    bundled_python_tree,
    cpu_onnxruntime_shadows_gpu,
    default_model_component_id,
    ensure_default_model,
    ensure_runtime,
    managed_ids,
    read_manifest,
    release_runtime,
)

UNITS = ("python", "gpu")


def _hooks() -> tuple[list[str], dict[str, object], dict[str, object], dict[str, object]]:
    calls: list[str] = []

    def probe(name: str):
        def check(path: Path) -> bool:
            return (path / name).is_file()

        return check

    def install(name: str):
        def run(path: Path, progress) -> None:
            calls.append(name)
            progress(f"Installing {name}")
            (path / name).write_text("ready", encoding="utf-8")

        return run

    def remove(name: str):
        def run(path: Path) -> None:
            (path / name).unlink(missing_ok=True)

        return run

    probes = {name: probe(name) for name in UNITS}
    installers = {name: install(name) for name in UNITS}
    removers = {name: remove(name) for name in UNITS}
    return calls, probes, installers, removers


def test_shared_dir_follows_voxprint_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    assert shared_dir() == tmp_path / "home" / "shared"


def test_ci_skips_without_installing(tmp_path: Path) -> None:
    def boom(_path: Path, _progress) -> None:
        raise AssertionError("installer ran")

    report = ensure_runtime("enlarge", root=tmp_path, installers={name: boom for name in UNITS})
    assert report.ok is True
    assert report.skipped is True
    assert {item.id for item in report.components} == set(managed_ids())
    assert not (tmp_path / "manifest.json").exists()


def test_holders_are_shared_and_removed_only_at_zero(tmp_path: Path) -> None:
    calls, probes, installers, removers = _hooks()
    kwargs = {"root": tmp_path, "allow_skip": False, "probes": probes, "installers": installers}
    first = ensure_runtime("enlarge", **kwargs)
    assert first.ok is True
    assert calls == list(UNITS)
    again = ensure_runtime("enlarge", **kwargs)
    assert calls == list(UNITS)
    assert {item.holders for item in again.components} == {("enlarge",)}
    clarity = ensure_runtime("clarity", **kwargs)
    assert calls == list(UNITS)
    assert {item.holders for item in clarity.components} == {("enlarge", "clarity")}
    released = release_runtime("enlarge", root=tmp_path, removers=removers)
    assert released.ok is True
    assert all((tmp_path / name).is_file() for name in UNITS)
    assert {item.holders for item in released.components} == {("clarity",)}
    gone = release_runtime("clarity", root=tmp_path, removers=removers)
    assert gone.ok is True
    assert {item.detail for item in gone.components} == {"removed"}
    assert all(not (tmp_path / name).exists() for name in UNITS)
    manifest = read_manifest(tmp_path)
    assert manifest["components"] == {}


def test_a_missing_component_is_repaired_once(tmp_path: Path) -> None:
    calls, probes, installers, removers = _hooks()
    ensure_runtime("enlarge", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    (tmp_path / "gpu").unlink()
    calls.clear()
    report = ensure_runtime("enlarge", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    assert calls == ["gpu"]
    wheels = [item for item in report.components if item.id in {"torch-cu130", "onnxruntime-gpu-1.31", "tensorrt-10"}]
    assert len(wheels) == 3
    assert {item.state for item in wheels} == {"repaired"}
    assert {item.holders for item in wheels} == {("enlarge",)}
    assert removers["gpu"](tmp_path) is None


def test_a_version_change_reinstalls_that_component(tmp_path: Path) -> None:
    calls, probes, installers, _removers = _hooks()
    ensure_runtime("enlarge", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["components"]["python-3.14"]["version"] = "3.13"
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls.clear()
    report = ensure_runtime("enlarge", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    assert calls == ["python"]
    python = next(item for item in report.components if item.id == "python-3.14")
    assert python.version == PYTHON_VERSION
    assert python.state == "repaired"
    assert read_manifest(tmp_path)["components"]["python-3.14"]["version"] == PYTHON_VERSION


def test_a_failed_install_does_not_take_a_hold(tmp_path: Path) -> None:
    def fail(_path: Path, _progress) -> None:
        raise SuiteInstallError("disk full")

    probes = {name: (lambda _path: False) for name in UNITS}
    installers = {name: fail for name in UNITS}
    report = ensure_runtime("enlarge", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    assert report.ok is False
    assert all(item.state == "failed" for item in report.components)
    assert read_manifest(tmp_path)["components"] == {}


def test_same_app_is_not_counted_twice_and_bad_ids_are_refused(tmp_path: Path) -> None:
    _calls, probes, installers, _removers = _hooks()
    ensure_runtime("enlarge", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    with pytest.raises(SuiteInstallError):
        ensure_runtime("../enlarge", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    assert read_manifest(tmp_path)["components"]["python-3.14"]["refs"] == ["enlarge"]


def test_python_copy_and_bundle_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "app" / "python" / "bin"
    bundle.mkdir(parents=True)
    exe = bundle / "python3"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr("enlarge.suite_runtime.bundled_python_tree", lambda: bundle.parent)
    monkeypatch.setattr(
        "enlarge.suite_runtime._python_ready",
        lambda path, _rel="python": (path / "python" / "bin" / "python3").is_file(),
    )
    root = tmp_path / "shared"

    def ready_gpu(_path: Path, _rel: str = "python") -> bool:
        return True

    def ready_model() -> bool:
        return True

    def skip_gpu(_path: Path, _rel: str, _progress) -> None:
        return None

    def skip_model(_progress) -> None:
        return None

    monkeypatch.setattr("enlarge.suite_runtime._install_gpu", skip_gpu)
    monkeypatch.setattr("enlarge.suite_runtime._install_model", skip_model)
    monkeypatch.setattr("enlarge.suite_runtime._gpu_ready", ready_gpu)
    monkeypatch.setattr("enlarge.suite_runtime._model_ready", ready_model)
    report = ensure_runtime("enlarge", root=root, allow_skip=False)
    assert report.ok is True
    copied = root / "python" / "bin" / "python3"
    assert copied.is_file()
    assert copied.read_text(encoding="utf-8").startswith("#!/bin/sh")
    if sys.platform != "win32":
        monkeypatch.setattr("sys.executable", str(exe))
        assert bundled_python_tree() == bundle.parent


def test_release_keeps_an_unknown_component_another_app_holds(tmp_path: Path) -> None:
    _calls, probes, installers, removers = _hooks()
    ensure_runtime("enlarge", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["components"]["future"] = {"holders": ["enlarge", "clarity"], "version": "1"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    release_runtime("enlarge", root=tmp_path, removers=removers)
    future = read_manifest(tmp_path)["components"]["future"]
    assert future["refs"] == ["clarity"]
    assert "holders" not in future


def test_selftest_json_and_uninstall_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--selftest", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["skipped"] is True
    assert payload["tensorrt_ep"]["detail"] == "skipped"
    assert payload["cuda_ep"]["detail"] == "skipped"
    assert {item["id"] for item in payload["components"]} == set(managed_ids()) | {default_model_component_id()}
    assert payload["voxprint_home_source"] in {"env", "default"}
    assert payload["voxprint_home"]
    assert main(["--install-suite"]) == 0
    install_out = capsys.readouterr().out
    assert "shared suite: skipped" in install_out
    assert "Voxprint home:" in install_out
    assert "(env)" in install_out
    assert main(["--uninstall-suite"]) == 0
    assert main(["--selftest"]) == 0


def test_selftest_requires_tensorrt_and_cuda(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)

    def ready(_app: str, **_kwargs: object) -> RuntimeReport:
        return RuntimeReport(True, False, ())

    monkeypatch.setattr("main.ensure_runtime", ready)
    monkeypatch.setattr("main.ensure_default_model", ready)
    monkeypatch.setattr("main.available_providers", lambda: ["CUDAExecutionProvider"])
    assert main(["--selftest", "--json"]) == 1
    failed = capsys.readouterr()
    payload = json.loads(failed.out)
    assert payload["ok"] is False
    assert payload["tensorrt_ep"]["ok"] is False
    assert payload["cuda_ep"]["ok"] is True
    assert "TensorrtExecutionProvider did not load" in failed.err

    monkeypatch.setattr(
        "main.available_providers",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    )
    assert main(["--selftest", "--json"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["ok"] is True
    assert loaded["tensorrt_ep"]["detail"] == "loaded"
    assert loaded["cuda_ep"]["detail"] == "loaded"


def test_selftest_fails_when_the_cpu_wheel_shadows_the_gpu_build(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)

    def ready(_app: str, **_kwargs: object) -> RuntimeReport:
        return RuntimeReport(True, False, ())

    monkeypatch.setattr("main.ensure_runtime", ready)
    monkeypatch.setattr("main.ensure_default_model", ready)
    monkeypatch.setattr(
        "main.available_providers",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    )
    monkeypatch.setattr(
        "main.cpu_onnxruntime_shadows_gpu",
        lambda: "CPU onnxruntime is installed beside onnxruntime-gpu, so the providers are Azure and CPU only.",
    )
    assert main(["--selftest", "--json"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "CPU onnxruntime is installed beside onnxruntime-gpu" in captured.err


def test_cpu_wheel_beside_the_gpu_build_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Dist:
        def __init__(self, name: str) -> None:
            self.metadata = {"Name": name}

    monkeypatch.setattr(
        "enlarge.suite_runtime.importlib_metadata.distributions",
        lambda: [_Dist("onnxruntime")],
    )
    assert cpu_onnxruntime_shadows_gpu() is None
    monkeypatch.setattr(
        "enlarge.suite_runtime.importlib_metadata.distributions",
        lambda: [_Dist("onnxruntime"), _Dist("onnxruntime-gpu")],
    )
    message = cpu_onnxruntime_shadows_gpu()
    assert message is not None
    assert "CPU onnxruntime" in message
    assert "onnxruntime-gpu" in message


def test_selftest_failure_is_nonzero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)

    def broken(_app: str, **_kwargs) -> RuntimeReport:
        return RuntimeReport(False, False, (ComponentStatus("gpu", "cu130", "failed", (), "pip failed"),))

    monkeypatch.setattr("main.ensure_runtime", broken)
    monkeypatch.setattr(
        "main.ensure_default_model",
        lambda *_args, **_kwargs: RuntimeReport(True, False, ()),
    )
    assert main(["--selftest", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["components"][0]["detail"] == "pip failed"


def test_default_model_id_is_the_manifest_version() -> None:
    assert DEFAULT_MODEL_ID == "4x-nomos2-realplksr-dysample"


def test_weight_file_is_removed_only_when_the_last_app_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(models))
    model = load_builtin(release=True).get(DEFAULT_MODEL_ID)
    assert model is not None
    target = models / model.filename
    target.parent.mkdir(parents=True)
    target.write_bytes(b"weight")
    register("enlarge")
    register("clarity")
    root = tmp_path / "shared"
    root.mkdir()
    payload = {
        "schema": 1,
        "components": {"model": {"holders": ["enlarge", "clarity"], "version": DEFAULT_MODEL_ID}},
    }
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert release_runtime("enlarge", root=root).ok is True
    assert target.is_file()
    assert release_runtime("clarity", root=root).ok is True
    assert not target.exists()


def test_installed_components_record_the_caller_and_skip_the_default_model(tmp_path: Path) -> None:
    calls, probes, installers, _removers = _hooks()

    def model_install(_path: Path, _progress) -> None:
        raise AssertionError("ensure_runtime installed the default model")

    installers = dict(installers)
    installers["model"] = model_install
    report = ensure_runtime("canvas", root=tmp_path, allow_skip=False, probes=probes, installers=installers)
    assert report.ok is True
    assert calls == ["python", "gpu"]
    components = read_manifest(tmp_path)["components"]
    assert components
    assert all(not str(key).startswith("model-") for key in components)
    for entry in components.values():
        assert entry["kind"] in {"runtime", "wheels"}
        assert "canvas" in entry["refs"]


def test_default_model_download_is_enlarge_only(tmp_path: Path) -> None:
    calls: list[str] = []

    def install(path: Path, progress) -> None:
        calls.append("model")
        progress("Installing model")
        (path / "model").write_text("ready", encoding="utf-8")

    def probe(path: Path) -> bool:
        return (path / "model").is_file()

    hooks = {"probes": {"model": probe}, "installers": {"model": install}, "root": tmp_path, "allow_skip": False}
    skipped = ensure_default_model("clarity", **hooks)
    assert calls == []
    assert skipped.ok is True
    assert skipped.skipped is True
    assert skipped.components[0].detail == "not this app"
    assert read_manifest(tmp_path)["components"] == {}
    installed = ensure_default_model("enlarge", **hooks)
    assert calls == ["model"]
    assert installed.ok is True
    components = read_manifest(tmp_path)["components"]
    assert list(components) == [default_model_component_id()]
    assert components[default_model_component_id()]["kind"] == "model"
    assert components[default_model_component_id()]["refs"] == ["enlarge"]


def test_windows_bundle_python_is_the_python_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "python"
    folder.mkdir()
    exe = folder / "python.exe"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("sys.executable", str(exe))
    assert bundled_python_tree() == folder.resolve()
