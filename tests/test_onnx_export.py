"""ONNX conversion: legacy exporter, sha256 cache, and a prebuilt source."""
from __future__ import annotations

import hashlib
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from voxprint_engine.onnx_export import OnnxExportError, ensure_onnx, export_module, onnx_cache_path
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.suite_runtime import _GPU_PROBE, _model_ready, install_gpu_wheels


def _manifest(**overrides: object) -> ModelManifest:
    data: dict[str, object] = {
        "id": "4x-example-model",
        "name": "Example",
        "arch": "rrdb",
        "scale": 4,
        "role": "upscale",
        "licence": "MIT",
        "author": "Someone",
        "attribution": "Example by Someone, MIT.",
        "source_url": "https://example.com/model",
        "sha256": "a" * 64,
        "vram_mb": 1000,
        "tile": 256,
        "filename": "example.pth",
        "summary": "A test manifest.",
        "selectable": True,
        "download_url": "https://example.com/example.pth",
    }
    data.update(overrides)
    return ModelManifest(**data)  # type: ignore[arg-type]


def _nearest(path: Path) -> None:
    inputs = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "H", "W"])
    outputs = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H2", "W2"])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "identity", [inputs], [outputs])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_legacy_exporter_requests_dynamo_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    captured: dict[str, object] = {}

    def fake_export(_module: object, _example: object, _dest: str, **kwargs: object) -> None:
        captured.update(kwargs)
        raise RuntimeError("stop after recording the exporter arguments")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    with pytest.raises(OnnxExportError, match="stop after recording"):
        export_module(torch.nn.Conv2d(3, 3, 1), tmp_path / "tiny.onnx", height=8, width=8)
    assert captured["dynamo"] is False
    assert captured["opset_version"] == 17


def test_tiny_export_passes_the_checker(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dest = tmp_path / "tiny.onnx"
    export_module(torch.nn.Conv2d(3, 3, 1), dest, height=8, width=8)
    model = onnx.load(dest)
    onnx.checker.check_model(model)
    names = [node.name for node in model.graph.node]
    assert names
    assert len(names) == len(set(names))
    assert TensorProto.FLOAT16 in {item.data_type for item in model.graph.initializer}


def test_four_channel_export_sets_the_graph_depth(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dest = tmp_path / "four.onnx"
    export_module(torch.nn.Conv2d(4, 4, 1), dest, height=8, width=8, channels=4)
    model = onnx.load(dest)
    onnx.checker.check_model(model)
    dims = model.graph.input[0].type.tensor_type.shape.dim
    assert dims[1].dim_value == 4
    with pytest.raises(OnnxExportError, match="in_channels"):
        export_module(torch.nn.Conv2d(3, 3, 1), tmp_path / "bad.onnx", channels=0)


def test_ensure_onnx_uses_the_manifest_channel_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"weight-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    root = tmp_path / "models"
    root.mkdir()
    weight = root / "example.pth"
    weight.write_bytes(payload)
    seen: list[int] = []

    def convert(src: Path, dest: Path, on_progress: object = None, *, channels: int = 3) -> None:
        del src, on_progress
        seen.append(channels)
        _nearest(dest)

    monkeypatch.setattr("voxprint_engine.onnx_export.export_weight", convert)
    path = ensure_onnx(_manifest(sha256=digest, in_channels=4), root, weight=weight)
    assert seen == [4]
    assert path.is_file()


def test_conversion_is_cached_by_weight_sha256(tmp_path: Path) -> None:
    payload = b"weight-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    root = tmp_path / "models"
    root.mkdir()
    weight = root / "example.pth"
    weight.write_bytes(payload)
    good = tmp_path / "good.onnx"
    _nearest(good)
    calls: list[Path] = []

    def convert(src: Path, dest: Path) -> None:
        calls.append(src)
        dest.write_bytes(good.read_bytes())

    model = _manifest(sha256=digest)
    notes: list[str] = []
    path = ensure_onnx(model, root, lambda _fraction, message: notes.append(message), converter=convert)
    assert path == onnx_cache_path(digest, root)
    assert calls == [weight]
    assert any("converting" in note for note in notes)
    again = ensure_onnx(model, root, converter=convert)
    assert again == path
    assert calls == [weight]


def test_a_checked_sibling_is_adopted(tmp_path: Path) -> None:
    payload = b"weight-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    root = tmp_path / "models"
    root.mkdir()
    weight = root / "example.pth"
    weight.write_bytes(payload)
    _nearest(weight.with_suffix(".onnx"))

    def convert(_src: Path, _dest: Path) -> None:
        raise AssertionError("a valid sibling should be cached without a new export")

    path = ensure_onnx(_manifest(sha256=digest), root, converter=convert)
    assert path == root / f"{digest}.onnx"
    assert path.is_file()


def test_prebuilt_onnx_is_preferred(tmp_path: Path) -> None:
    good = tmp_path / "mirror.onnx"
    _nearest(good)
    payload = good.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    root = tmp_path / "models"
    fetched: list[str] = []

    def fetch(url: str, target: Path, expected: str, _progress: object) -> None:
        fetched.append(url)
        assert expected == digest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def convert(_src: Path, _dest: Path) -> None:
        raise AssertionError("a prebuilt ONNX file skips conversion")

    model = _manifest(onnx_url="https://example.com/mirror.onnx", onnx_sha256=digest)
    path = ensure_onnx(model, root, converter=convert, downloader=fetch)
    assert fetched == ["https://example.com/mirror.onnx"]
    assert path.name == f"{digest}.onnx"
    ensure_onnx(model, root, converter=convert, downloader=fetch)
    assert fetched == ["https://example.com/mirror.onnx"]


def test_gpu_probe_rejects_the_cpu_wheel_and_requires_tensorrt_10_14() -> None:
    assert "10.14.1.48" in _GPU_PROBE
    assert 'if "onnxruntime" in names' in _GPU_PROBE
    assert "cpu-onnxruntime" in _GPU_PROBE
    compile(_GPU_PROBE, "<gpu-probe>", "exec")


def test_gpu_install_includes_the_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_pip(_python: str, args: list[str], _message: str, _progress: object, *, required: bool) -> None:
        del required
        calls.append(args)

    monkeypatch.setattr("voxprint_engine.suite_runtime._pip", fake_pip)
    install_gpu_wheels("python")
    flat = " ".join(" ".join(args) for args in calls)
    assert "spandrel>=0.4,<0.5" in flat
    assert "onnx>=1.16,<2" in flat
    assert "onnxruntime-gpu==1.31.0" in flat
    assert "--extra-index-url https://pypi.nvidia.com" in flat
    assert "tensorrt-cu13-libs==10.14.1.48" in flat
    assert "--no-deps tensorrt-cu13==10.14.1.48" in flat
    assert "uninstall -y onnxruntime " in flat
    assert "torch" in flat


def test_model_probe_requires_the_onnx_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "ab" * 32
    monkeypatch.setattr("voxprint_engine.suite_runtime.weight_report", lambda _model: ("ready", "ok"))
    monkeypatch.setattr("voxprint_engine.suite_runtime._default_model", lambda: _manifest(sha256=digest))
    monkeypatch.setattr("voxprint_engine.onnx_export.models_dir", lambda: tmp_path)
    assert _model_ready() is False
    (tmp_path / f"{digest}.onnx").write_bytes(b"cached")
    assert _model_ready() is True
