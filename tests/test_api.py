"""The names Clarity is allowed to import. Additions update this list in the same pull request."""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import voxprint_engine
import voxprint_engine.api as api

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
    assert voxprint_engine.__version__ == "1.2.0"
    assert api.__version__ == "1.2.0"


def test_public_names_match_the_contract() -> None:
    assert tuple(api.__all__) == CONTRACT
    for name in CONTRACT:
        assert getattr(api, name) is not None


def test_distribution_is_the_engine_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "voxprint-engine"
    assert project["project"]["version"] == "1.2.0"
    assert project["project"]["requires-python"] == ">=3.14,<3.15"
    assert "psutil>=5.9,<8" in project["project"]["dependencies"]
    include = project["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == ["voxprint_engine*"]
    text = (ROOT / "docs" / "ENGINE.md").read_text(encoding="utf-8")
    assert "voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0" in text


def test_api_module_only_reexports() -> None:
    """The public module assigns imports. It does not grow a second implementation."""
    tree = ast.parse((ROOT / "voxprint_engine" / "api.py").read_text(encoding="utf-8"))
    for node in tree.body:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr, ast.AnnAssign))
