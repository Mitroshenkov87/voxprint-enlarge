"""Tiling, the provider order, the model store, and a CPU ONNX smoke test."""
from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

from voxprint_engine.errors import ImageJobError
from voxprint_engine.formats import open_rgb
from voxprint_engine.infer import _passes, _run_network, upscale_file, uses_lanczos
from voxprint_common.paths import models_dir, trt_cache_dir
from voxprint_engine.providers import CPU, CUDA, TRT, choices_for, cuda_fallback_choices, gpu_provider_names
from voxprint_engine.registry.schema import ModelManifest
from voxprint_engine.store import (
    StoreError,
    _SameRegistryRedirect,
    download,
    ensure_weight,
    register,
    unregister,
)
from voxprint_engine.tile import blend, feather, plan_tiles, upscale_tiled
from voxprint_engine.vram import halve_tile, is_memory_error


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


def test_tiles_cover_every_pixel_and_stay_inside() -> None:
    boxes = plan_tiles(100, 40, tile=32, overlap=8)
    assert boxes
    for box in boxes:
        assert box.x >= 0 and box.y >= 0
        assert box.x + box.width <= 100
        assert box.y + box.height <= 40
    covered = np.zeros((40, 100), dtype=np.uint8)
    for box in boxes:
        covered[box.y:box.y + box.height, box.x:box.x + box.width] = 1
    assert int(covered.min()) == 1


def test_feather_on_a_shared_edge_sums_to_one() -> None:
    overlap = 8
    left = feather(16, 32, overlap, fade_left=False, fade_right=True, fade_top=False, fade_bottom=False)
    right = feather(16, 32, overlap, fade_left=True, fade_right=False, fade_top=False, fade_bottom=False)
    assert np.allclose(left[:, -overlap:] + right[:, :overlap], 1.0)
    assert np.allclose(left[:, 0], 1.0)
    assert np.allclose(right[:, -1], 1.0)


def test_blend_of_one_full_weight_keeps_the_image() -> None:
    image = np.full((4, 4, 3), 0.25, dtype=np.float32)
    weight = np.ones((4, 4), dtype=np.float32)
    merged = blend(4, 4, 3, [(0, 0, image, weight)])
    assert np.allclose(merged, image)


def test_identity_tiles_reconstruct_the_image() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((20, 30, 3), dtype=np.float32)
    out = upscale_tiled(image, lambda tile: tile, tile=16, overlap=4, scale=1)
    assert out.shape == image.shape
    assert np.allclose(out, image, atol=1e-5)


def test_constant_image_stays_constant_when_scaled() -> None:
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)

    def twice(tile: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

    out = upscale_tiled(image, twice, tile=8, overlap=0, scale=2)
    assert out.shape == (16, 16, 3)
    assert np.allclose(out, 0.5)


def test_native_passes_then_downscale() -> None:
    assert _passes(2, 4) == 1
    assert _passes(4, 4) == 1
    assert _passes(16, 4) == 2
    assert halve_tile(64) == 32
    assert halve_tile(32) is None
    assert is_memory_error(RuntimeError("CUDA out of memory"))
    assert not is_memory_error(ValueError("bad image"))


def test_oom_halves_the_tile_and_retries() -> None:
    seen: list[int] = []

    def infer(tile: np.ndarray) -> np.ndarray:
        seen.append(tile.shape[0])
        if tile.shape[0] > 32:
            raise RuntimeError("failed to allocate")
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

    image = np.zeros((8, 8, 3), dtype=np.float32)
    produced, tile, spilled = _run_network(image, infer, 64, 2, 2)
    assert spilled
    assert tile == 32
    assert seen == [64, 32]
    assert produced.shape == (16, 16, 3)


def test_provider_order_is_tensorrt_then_cuda_after_a_failed_build() -> None:
    available = [TRT, CUDA, "DmlExecutionProvider", "WinMLExecutionProvider", CPU]
    assert gpu_provider_names(available, device="auto") == [TRT, CUDA]
    assert [choice.name for choice in choices_for(available, "auto")] == [TRT]
    assert [choice.name for choice in cuda_fallback_choices(available, "auto")] == [CUDA]
    assert [choice.name for choice in choices_for([CUDA, "DmlExecutionProvider"], "auto")] == [CUDA]
    assert cuda_fallback_choices([CUDA, "DmlExecutionProvider"], "auto") == []
    assert gpu_provider_names(["DmlExecutionProvider", "WinMLExecutionProvider"], device="auto") == []
    assert choices_for(["DmlExecutionProvider"], "auto") == []
    assert [choice.name for choice in choices_for(available, "auto", execution="cuda")] == [CUDA]
    assert [choice.name for choice in choices_for(available, "auto", execution="tensorrt")] == [TRT]
    assert choices_for([TRT], "auto", execution="cuda") == []
    assert gpu_provider_names(available, device="cpu") == []
    assert gpu_provider_names([CPU], device="auto") == []


def test_tensorrt_profile_uses_in_channels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_TRT_CACHE", str(tmp_path / "cache"))
    choice = choices_for([TRT], "auto", in_channels=4)[0]
    assert choice.options["trt_profile_min_shapes"] == "input:1x4x8x8"
    assert choice.options["trt_profile_opt_shapes"] == "input:1x4x256x256"
    assert choice.options["trt_profile_max_shapes"] == "input:1x4x2048x2048"
    default = choices_for([TRT], "auto")[0]
    assert default.options["trt_profile_opt_shapes"] == "input:1x3x256x256"
    with pytest.raises(ValueError, match="in_channels"):
        choices_for([TRT], "auto", in_channels=0)


def test_trt_options_use_the_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_TRT_CACHE", str(tmp_path / "cache"))
    choice = choices_for([TRT], "cuda:1", gpu="RTX 4090", driver="560.70", trt="10.0.1")[0]
    assert choice.name == TRT
    assert choice.options["trt_fp16_enable"] == "True"
    assert choice.options["trt_engine_cache_enable"] == "True"
    assert choice.options["device_id"] == "1"
    folder = Path(choice.options["trt_engine_cache_path"])
    assert folder.is_dir()
    assert folder == trt_cache_dir()


def test_cpu_onnx_mode_does_not_ask_for_tensorrt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_ORT_DEVICE", "cpu")
    monkeypatch.setenv("VOXPRINT_TRT_CACHE", str(tmp_path / "cache"))
    choices = choices_for([TRT, CUDA], "auto")
    assert [choice.name for choice in choices] == [CPU]
    assert not (tmp_path / "cache").exists()


def test_models_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path / "weights"))
    assert models_dir() == tmp_path / "weights"


class _Body:
    def __init__(self, payload: bytes, url: str) -> None:
        self._payload = payload
        self._url = url
        self.headers = {"Content-Length": str(len(payload))}
        self.status = 200

    def read(self, size: int) -> bytes:
        chunk = self._payload[:size]
        self._payload = self._payload[size:]
        return chunk

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def test_download_checks_https_hash_and_users(tmp_path: Path) -> None:
    root = tmp_path / "models"
    payload = b"weight-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    target = root / "example.pth"
    url = "https://huggingface.co/voxprint/models/resolve/main/example.pth"
    with pytest.raises(StoreError, match="https"):
        download("http://example.com/example.pth", target, digest)
    with pytest.raises(StoreError, match="registry host"):
        download("https://example.com/example.pth", target, digest)
    wrong = b"not-the-weight"
    public = mock.patch("voxprint_engine.store._addresses", return_value=["1.1.1.1"])
    with public, mock.patch("voxprint_engine.store._open_registry", side_effect=lambda *_args, **_kwargs: _Body(wrong, url)):
        with pytest.raises(StoreError, match="does not match"):
            download(url, target, digest)
    assert not target.exists()
    with public, mock.patch("voxprint_engine.store._open_registry", side_effect=lambda *_args, **_kwargs: _Body(payload, url)):
        download(url, target, digest)
    assert target.read_bytes() == payload
    model = _manifest(sha256=digest)
    assert ensure_weight(model, root) == target
    target.write_bytes(b"tampered")
    with pytest.raises(StoreError, match="sha256"):
        ensure_weight(model, root)
    assert target.read_bytes() == b"tampered"
    blocked = _manifest(id="4x-blocked", licence="CC-BY-NC-4.0", filename="blocked.pth")
    with pytest.raises(StoreError, match="not licensed"):
        ensure_weight(blocked, root)
    assert not (root / "blocked.pth").exists()
    (root / ".users.json").write_text('{"audiobook-builder": true, "movie-dubber": true}', encoding="utf-8")
    users = register("enlarge", root)
    assert users["audiobook-builder"] is True
    assert users["enlarge"] is True
    again = register("enlarge", root)
    assert again == users
    assert unregister("enlarge", root) == ["audiobook-builder", "movie-dubber"]


def test_download_refuses_private_addresses_and_cross_registry_redirects(tmp_path: Path) -> None:
    url = "https://github.com/Phhofm/models/releases/download/v1/model.pth"
    digest = "ab" * 32
    target = tmp_path / "model.pth"
    for address in ("127.0.0.1", "10.1.2.3", "169.254.169.254", "::1", "::ffff:127.0.0.1"):
        with mock.patch("voxprint_engine.store._addresses", return_value=[address]):
            with pytest.raises(StoreError, match="private"):
                download(url, target, digest)
    handler = _SameRegistryRedirect()
    request = urllib.request.Request(url)
    with mock.patch("voxprint_engine.store._addresses", return_value=["1.1.1.1"]):
        with pytest.raises(StoreError, match="another host"):
            handler.redirect_request(request, None, 302, "Found", {}, "https://evil.example/model.pth")
        with pytest.raises(StoreError, match="another host"):
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://huggingface.co/voxprint/models/resolve/main/model.pth",
            )
        allowed = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com/github-production-release-asset/model.pth",
        )
    assert allowed is not None
    assert urllib.parse.urlsplit(allowed.full_url).hostname == "release-assets.githubusercontent.com"


def test_converter_does_not_import_extra_arches() -> None:
    paths = list(Path("voxprint_engine").rglob("*.py"))
    paths.append(Path("tools/export_onnx.py"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "import spandrel_extra_arches" not in text
        assert "from spandrel_extra_arches" not in text


def _nearest_x2(path: Path) -> None:
    inputs = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "H", "W"])
    outputs = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H2", "W2"])
    roi = helper.make_tensor("roi", TensorProto.FLOAT, [0], [])
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
    node = helper.make_node(
        "Resize",
        ["input", "roi", "scales"],
        ["output"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
    )
    graph = helper.make_graph([node], "nearest_x2", [inputs], [outputs], [roi, scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_a_job_without_a_gpu_provider_does_not_resample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    monkeypatch.setattr("voxprint_engine.infer.gpu_provider_names", lambda *_args, **_kwargs: [])
    src = tmp_path / "in.png"
    dst = tmp_path / "out.avif"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(src)
    with pytest.raises(ImageJobError, match="NVIDIA GPU"):
        upscale_file(
            src,
            dst,
            _manifest(),
            scale=2,
            image_format="avif",
            quality=60,
            double=False,
            device="auto",
            user_tile=0,
            batch=1,
        )
    assert not dst.exists()


def test_ci_flag_is_the_only_resample(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _manifest()
    monkeypatch.setenv("VOXPRINT_CI_LANCZOS", "1")
    assert uses_lanczos(model, "auto", "auto")
    assert not uses_lanczos(model, "cpu-onnx", "onnx")
    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    assert not uses_lanczos(model, "cpu", "auto")
    assert not uses_lanczos(model, "auto", "lanczos")


def test_cpu_onnx_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    onnx_path = tmp_path / "nearest.onnx"
    _nearest_x2(onnx_path)
    src = tmp_path / "in.png"
    dst = tmp_path / "out.jxl"
    color = (10, 20, 30)
    Image.new("RGB", (8, 8), color).save(src)
    model = _manifest(id="lanczos-not-this", arch="rrdb", scale=2, filename="nearest.pth", download_url="")
    started = time.perf_counter()
    outcome = upscale_file(
        src,
        dst,
        model,
        scale=2,
        image_format="jxl",
        quality=60,
        lossless=True,
        double=False,
        device="cpu-onnx",
        user_tile=32,
        batch=1,
        engine="onnx",
        onnx_path=onnx_path,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(f"CPU_ONNX_SMOKE_MS {elapsed_ms:.1f} engine={outcome.engine} message={outcome.message}")
    assert outcome.engine == "onnx"
    assert "CPUExecutionProvider" in outcome.message
    assert (outcome.width, outcome.height) == (16, 16)
    image = open_rgb(dst)
    assert image.size == (16, 16)
    assert image.getpixel((0, 0)) == color
    assert image.getpixel((15, 15)) == color
    captured = capsys.readouterr()
    assert "CPU_ONNX_SMOKE_MS" in captured.out
