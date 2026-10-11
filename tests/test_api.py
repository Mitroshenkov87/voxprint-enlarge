"""The names Clarity is allowed to import. Additions update this list in the same pull request."""
from __future__ import annotations

import ast
import importlib.util
import re
import tomllib
from pathlib import Path

import voxprint_engine
import voxprint_engine.api as api
from enlarge.suite_runtime import GPU_EXTRA

ROOT = Path(__file__).resolve().parents[1]

# Keep this tuple aligned with docs/ENGINE.md (Public API) and voxprint_engine.api.__all__.
CONTRACT = (
    "API",
    "Catalog",
    "DEFAULT_MODEL_ID",
    "LANCZOS_MODEL_ID",
    "ManifestError",
    "ModelManifest",
    "RestoreBatchRequest",
    "UpscaleRequest",
    "UpscaleResult",
    "WorkerClient",
    "WorkerError",
    "__version__",
    "choose_tile",
    "combined_catalog",
    "ensure_onnx",
    "ensure_weight",
    "halve_tile",
    "iter_tile_frames",
    "load_builtin",
    "note_spill",
    "read_shared_usage",
    "register",
    "restore_frames",
    "sysmem_spill",
    "temporal_windows",
    "tile_frames",
    "unregister",
    "upscale_file",
    "uses_lanczos",
    "validate_manifest",
    "vram_budget_gb",
)


def test_api_number_and_version() -> None:
    assert voxprint_engine.API == 1
    assert api.API == 1
    assert voxprint_engine.__version__ == "1.4.0"
    assert api.__version__ == "1.4.0"


def test_public_names_match_the_contract() -> None:
    assert tuple(api.__all__) == CONTRACT
    for name in CONTRACT:
        assert getattr(api, name) is not None


def test_suite_modules_are_not_in_the_engine() -> None:
    assert importlib.util.find_spec("voxprint_engine.shared_manifest") is None
    assert importlib.util.find_spec("voxprint_engine.ui") is None
    assert importlib.util.find_spec("voxprint_engine.paths") is None
    assert importlib.util.find_spec("voxprint_engine.gpu_lock") is None
    assert importlib.util.find_spec("voxprint_engine.hardware") is None
    assert importlib.util.find_spec("voxprint_engine.suite_runtime") is None
    assert not (ROOT / "handoff").exists()
    assert not (ROOT / "enlarge" / "_vendor").exists()
    runtime = (ROOT / "enlarge" / "suite_runtime.py").read_text(encoding="utf-8")
    assert "voxprint_common.runtime_manifest" in runtime
    assert "enlarge._vendor" not in runtime


def test_distribution_is_the_engine_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "voxprint-engine"
    assert project["project"]["version"] == "1.4.0"
    assert project["project"]["requires-python"] == ">=3.14,<3.15"
    assert any(spec.startswith("voxprint-common") for spec in project["project"]["dependencies"])
    assert "psutil" not in " ".join(project["project"]["dependencies"])
    include = project["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == ["voxprint_engine*"]
    text = (ROOT / "docs" / "ENGINE.md").read_text(encoding="utf-8")
    assert "voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0" in text


def _requirement_name(spec: str) -> str:
    text = spec.split("#", 1)[0].strip()
    if not text or text.startswith(("-", "--")):
        return ""
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", text)
    if match is None:
        return ""
    return match.group(1).lower().replace("_", "-")


def test_pyproject_declares_the_gpu_extra_and_not_the_cpu_wheel() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = list(project["dependencies"])
    extras = project["optional-dependencies"]
    for group in extras.values():
        declared.extend(group)
    names = [_requirement_name(spec) for spec in declared]
    assert "onnxruntime" not in names
    assert tuple(extras["gpu"]) == GPU_EXTRA
    for path in sorted(ROOT.glob("requirements*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert _requirement_name(line) != "onnxruntime"
    for path in sorted((ROOT / "tools").glob("requirements*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert _requirement_name(line) != "onnxruntime"


def test_api_module_only_reexports() -> None:
    """The public module assigns imports. It does not grow a second implementation."""
    tree = ast.parse((ROOT / "voxprint_engine" / "api.py").read_text(encoding="utf-8"))
    for node in tree.body:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr, ast.AnnAssign))
