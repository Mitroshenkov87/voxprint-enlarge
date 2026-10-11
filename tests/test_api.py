"""The names Clarity is allowed to import. Additions update this list in the same pull request."""
from __future__ import annotations

import ast
import importlib.util
import re
import tomllib
from pathlib import Path

import voxprint_engine
import voxprint_engine.api as api
from voxprint_engine.suite_runtime import GPU_EXTRA

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
    "ComponentStatus",
    "RuntimeReport",
    "SuiteInstallError",
    "decide",
    "ensure_onnx",
    "ensure_default_model",
    "ensure_runtime",
    "ensure_weight",
    "gpu_job",
    "halve_tile",
    "iter_tile_frames",
    "load_builtin",
    "models_dir",
    "note_spill",
    "read_manifest",
    "read_shared_usage",
    "release_runtime",
    "register",
    "restore_frames",
    "shared_dir",
    "sysmem_spill",
    "temporal_windows",
    "tile_frames",
    "trt_cache_dir",
    "unregister",
    "upscale_file",
    "uses_lanczos",
    "validate_manifest",
    "vram_budget_gb",
)


def test_api_number_and_version() -> None:
    assert voxprint_engine.API == 1
    assert api.API == 1
    assert voxprint_engine.__version__ == "1.2.1"
    assert api.__version__ == "1.2.1"


def test_public_names_match_the_contract() -> None:
    assert tuple(api.__all__) == CONTRACT
    for name in CONTRACT:
        assert getattr(api, name) is not None


def test_suite_modules_are_not_in_the_engine() -> None:
    assert importlib.util.find_spec("voxprint_engine.shared_manifest") is None
    assert importlib.util.find_spec("voxprint_engine.ui") is None
    header = "# vendored, to be replaced by the voxprint-common dependency\n"
    for name in ("manifest.py", "splash.py"):
        handoff = (ROOT / "handoff" / "voxprint_common" / name).read_text(encoding="utf-8")
        vendor = (ROOT / "enlarge" / "_vendor" / "voxprint_common" / name).read_text(encoding="utf-8")
        assert vendor == header + handoff
    runtime = (ROOT / "voxprint_engine" / "suite_runtime.py").read_text(encoding="utf-8")
    assert "handoff" not in runtime
    assert "enlarge._vendor.voxprint_common.manifest" in runtime


def test_distribution_is_the_engine_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "voxprint-engine"
    assert project["project"]["version"] == "1.2.1"
    assert project["project"]["requires-python"] == ">=3.14,<3.15"
    assert "psutil>=5.9,<8" in project["project"]["dependencies"]
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
