"""Shared suite runtime: one folder, one refcounted manifest.

Enlarge and Clarity both call ``ensure_runtime`` and ``release_runtime``.
The folder is ``%LOCALAPPDATA%\\Voxprint\\shared`` on Windows and
``~/.local/share/voxprint/shared`` elsewhere. ``manifest.json`` lists each
component, its version, and the apps that hold it. A payload is removed
only when that list is empty.

``VOXPRINT_CI_LANCZOS=1`` skips the network and reports every component as
skipped, so tests stay offline. Pass ``allow_skip=False`` to force the check.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from voxprint_engine.hardware import ci_lanczos
from voxprint_engine.paths import models_dir, shared_dir
from voxprint_engine.registry import DEFAULT_MODEL_ID, ModelManifest, load_builtin
from voxprint_engine.onnx_export import OnnxExportError, cached_onnx, ensure_onnx
from voxprint_engine.store import StoreError, register, unregister, weight_report

SCHEMA = 1
PYTHON_VERSION = "3.14"
GPU_VERSION = "cu130"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
ORT_GPU_SPEC = "onnxruntime-gpu==1.31.0"
TRT_LIBS_SPEC = "tensorrt-cu13-libs==10.14.1.48"
TRT_META_SPEC = "tensorrt-cu13==10.14.1.48"
NVIDIA_INDEX = "https://pypi.nvidia.com"
SPANDREL_SPEC = "spandrel>=0.4,<0.5"
ONNX_SPEC = "onnx>=1.16,<2"

ComponentId = Literal["python", "gpu", "model"]
Progress = Callable[[str], None]
Probe = Callable[[Path], bool]
Installer = Callable[[Path, Progress], None]
Remover = Callable[[Path], None]

_APP_ID = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_COMPONENTS: tuple[tuple[ComponentId, str], ...] = (
    ("python", PYTHON_VERSION),
    ("gpu", GPU_VERSION),
    ("model", DEFAULT_MODEL_ID),
)
_GPU_PROBE = """
import importlib.metadata as meta
import importlib.util as util
import sys
missing = [name for name in ("torch", "onnxruntime", "spandrel", "onnx") if util.find_spec(name) is None]
names = {
    dist.metadata["Name"].lower()
    for dist in meta.distributions()
    if dist.metadata and dist.metadata.get("Name")
}
if "onnxruntime" in names:
    missing.append("cpu-onnxruntime")
try:
    libs = meta.version("tensorrt-cu13-libs")
except meta.PackageNotFoundError:
    missing.append("tensorrt-cu13-libs")
else:
    if not str(libs).startswith("10.14.1.48"):
        missing.append("tensorrt-cu13-libs")
try:
    meta_version = meta.version("tensorrt-cu13")
except meta.PackageNotFoundError:
    missing.append("tensorrt-cu13")
else:
    if not str(meta_version).startswith("10.14.1.48"):
        missing.append("tensorrt-cu13")
for banned in ("tensorrt", "tensorrt-cu13-bindings"):
    try:
        meta.version(banned)
    except meta.PackageNotFoundError:
        continue
    missing.append(banned)
sys.exit(0 if not missing else 1)
"""


class SuiteInstallError(RuntimeError):
    """The shared runtime could not be checked, installed, or released."""


@dataclass(frozen=True)
class ComponentStatus:
    """One shared component after a check, a repair, or a release."""

    id: str
    version: str
    state: str
    holders: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict[str, object]:
        """JSON-ready fields."""
        return {
            "detail": self.detail,
            "holders": list(self.holders),
            "id": self.id,
            "state": self.state,
            "version": self.version,
        }


@dataclass(frozen=True)
class RuntimeReport:
    """Result of ``ensure_runtime`` or ``release_runtime``."""

    ok: bool
    skipped: bool
    components: tuple[ComponentStatus, ...]

    def as_dict(self) -> dict[str, object]:
        """JSON-ready fields. ``--selftest --json`` prints this object."""
        return {
            "components": [item.as_dict() for item in self.components],
            "ok": self.ok,
            "skipped": self.skipped,
        }


def read_manifest(root: Path | None = None) -> dict[str, object]:
    """Return the suite manifest. A missing or damaged file is an empty schema 1 document."""
    path = _manifest_path(root or shared_dir())
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return _empty_manifest()
    if not isinstance(data, dict):
        return _empty_manifest()
    schema = data.get("schema")
    if schema != SCHEMA:
        if schema is not None:
            raise SuiteInstallError(f"unsupported suite manifest schema {schema}")
        return _empty_manifest()
    raw = data.get("components")
    components: dict[str, dict[str, object]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, dict):
                components[key] = {
                    "holders": _holders(value.get("holders")),
                    "version": str(value.get("version") or ""),
                }
    return {"schema": SCHEMA, "components": components}


def ensure_runtime(
    app: str,
    *,
    repair: bool = True,
    progress: Progress | None = None,
    root: Path | None = None,
    probes: Mapping[str, Probe] | None = None,
    installers: Mapping[str, Installer] | None = None,
    allow_skip: bool = True,
) -> RuntimeReport:
    """Hold ``app`` on each shared component, installing anything that is missing.

    The same app a second time does not add a second holder. ``probes`` and
    ``installers`` replace the real checks for tests. The real steps copy or
    record Python 3.14, install the CUDA 13 wheels, and download the default model.
    """
    _check_app(app)
    if allow_skip and ci_lanczos():
        return _skipped_report()
    folder = root or shared_dir()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(folder)
    components = _components(manifest)
    statuses: list[ComponentStatus] = []
    changed = False
    for component, version in _COMPONENTS:
        entry = components.get(component)
        holders = list(_entry_holders(entry))
        files_ready = _call_probe(component, folder, probes)
        version_ok = entry is not None and str(entry.get("version") or "") == version
        ready = files_ready and (entry is None or version_ok)
        state = "ok"
        detail = "ready"
        _emit(progress, f"Checking {component} {version}")
        if not ready:
            if not repair:
                state = "missing"
                detail = "not installed"
            else:
                _emit(progress, f"Installing {component} {version}")
                try:
                    _call_installer(component, folder, progress, installers)
                except (SuiteInstallError, StoreError, OSError) as exc:
                    state = "failed"
                    detail = str(exc)
                else:
                    files_ready = _call_probe(component, folder, probes)
                    if files_ready:
                        state = "repaired"
                        detail = "repaired"
                        ready = True
                    else:
                        state = "failed"
                        detail = "still missing after install"
        if ready:
            if app not in holders:
                holders.append(app)
            new_entry: dict[str, object] = {"holders": holders, "version": version}
            if entry != new_entry:
                changed = True
            components[component] = new_entry
            if state == "ok":
                _emit(progress, f"{component} {version} is ready")
            if component == "model":
                _register_model_user(app)
        statuses.append(ComponentStatus(component, version, state, tuple(holders), detail))
    if changed:
        _write_manifest(manifest, folder)
    ok = all(item.state in {"ok", "repaired"} for item in statuses)
    return RuntimeReport(ok, False, tuple(statuses))


def release_runtime(
    app: str,
    *,
    progress: Progress | None = None,
    root: Path | None = None,
    removers: Mapping[str, Remover] | None = None,
) -> RuntimeReport:
    """Drop ``app`` from the manifest. Delete a payload only when its holder list is empty."""
    _check_app(app)
    folder = root or shared_dir()
    manifest = read_manifest(folder)
    components = _components(manifest)
    statuses: list[ComponentStatus] = []
    failed = False
    known = {component for component, _version in _COMPONENTS}
    for component, version in _COMPONENTS:
        entry = components.get(component)
        if entry is None:
            statuses.append(ComponentStatus(component, version, "ok", (), "absent"))
            continue
        previous = _entry_holders(entry)
        holders = [name for name in previous if name != app]
        if component == "model" and app in previous and holders:
            try:
                unregister(app)
            except (OSError, StoreError):
                pass
        if app in previous and not holders:
            _emit(progress, f"Removing {component} {version}")
            try:
                _call_remover(component, folder, app, removers)
            except (SuiteInstallError, StoreError, OSError) as exc:
                failed = True
                statuses.append(ComponentStatus(component, version, "failed", tuple(_entry_holders(entry)), str(exc)))
                continue
            del components[component]
            statuses.append(ComponentStatus(component, version, "ok", (), "removed"))
            continue
        if holders != _entry_holders(entry):
            updated: dict[str, object] = {"holders": holders, "version": str(entry.get("version") or version)}
            components[component] = updated
            _emit(progress, f"{component} {version} still used by {', '.join(holders)}")
        recorded = str(entry.get("version") or version)
        detail = "held" if holders else "absent"
        statuses.append(ComponentStatus(component, recorded, "ok", tuple(holders), detail))
    for key, entry in list(components.items()):
        if key in known or not isinstance(entry, dict):
            continue
        holders = [name for name in _entry_holders(entry) if name != app]
        if holders == _entry_holders(entry):
            continue
        if holders:
            components[key] = {"holders": holders, "version": str(entry.get("version") or "")}
        else:
            del components[key]
    if folder.exists():
        _write_manifest(manifest, folder)
    return RuntimeReport(not failed, False, tuple(statuses))


def install_gpu_wheels(python: str, progress: Progress | None = None) -> None:
    """Install torch for CUDA 13, ONNX Runtime GPU 1.31, and TensorRT 10.14.

    The CPU ``onnxruntime`` wheel shadows ``onnxruntime-gpu`` and leaves only
    the CPU and Azure providers. Unpinned ``tensorrt-cu13`` is TensorRT 11
    (``nvinfer_11``). ONNX Runtime 1.31 loads ``nvinfer_10`` from
    ``tensorrt-cu13-libs`` 10.14.1.48. Bindings for that TensorRT have no
    cp314 wheel, so the meta package is recorded with ``--no-deps``.
    """
    if getattr(sys, "frozen", False):
        raise SuiteInstallError("install the app directory, then run voxprint-enlarge --install-gpu")
    if not python:
        raise SuiteInstallError("a Python executable is required")
    _pip(python, ["install", "torch", "--index-url", TORCH_INDEX], "Installing torch for CUDA 13", progress, required=True)
    _pip(
        python,
        [
            "uninstall",
            "-y",
            "onnxruntime",
            "onnxruntime-gpu",
            "tensorrt",
            "tensorrt-cu13",
            "tensorrt-cu13-bindings",
            "tensorrt-cu13-libs",
        ],
        "Removing the CPU ONNX Runtime wheel and TensorRT 11",
        progress,
        required=False,
    )
    _pip(
        python,
        ["install", "--extra-index-url", NVIDIA_INDEX, ORT_GPU_SPEC, TRT_LIBS_SPEC, SPANDREL_SPEC, ONNX_SPEC],
        "Installing ONNX Runtime GPU 1.31 and TensorRT 10.14",
        progress,
        required=True,
    )
    _pip(
        python,
        ["uninstall", "-y", "onnxruntime"],
        "Removing the CPU ONNX Runtime wheel",
        progress,
        required=False,
    )
    _pip(
        python,
        ["install", "--no-deps", TRT_META_SPEC],
        "Recording the TensorRT 10.14 package without its bindings",
        progress,
        required=True,
    )


def bundled_python_tree() -> Path | None:
    """The ``python`` directory this process was started from, when it is a bundle."""
    executable = Path(sys.executable).resolve()
    parent = executable.parent
    if parent.name == "bin" and parent.parent.name == "python":
        return parent.parent
    if sys.platform == "win32" and parent.name == "python" and executable.name.lower() in {"python.exe", "pythonw.exe"}:
        return parent
    return None


def _skipped_report() -> RuntimeReport:
    items = tuple(
        ComponentStatus(component, version, "skipped", (), "CI stand-in") for component, version in _COMPONENTS
    )
    return RuntimeReport(True, True, items)


def _empty_manifest() -> dict[str, object]:
    return {"schema": SCHEMA, "components": {}}


def _components(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = manifest.get("components")
    if isinstance(raw, dict):
        return raw
    components: dict[str, dict[str, object]] = {}
    manifest["components"] = components
    return components


def _entry_holders(entry: dict[str, object] | None) -> list[str]:
    if entry is None:
        return []
    return _holders(entry.get("holders"))


def _holders(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: list[str] = []
    for item in value:
        name = str(item).strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _check_app(app: str) -> None:
    if _APP_ID.fullmatch(app) is None:
        raise SuiteInstallError("app id must be a short lowercase name")


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _emit(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _write_manifest(manifest: dict[str, object], root: Path) -> None:
    path = _manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"components": manifest.get("components", {}), "schema": SCHEMA}, indent=1, sort_keys=True) + "\n"
    tmp = path.with_name(f".manifest-{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp.exists():
            tmp.unlink()


def _call_probe(component: ComponentId, root: Path, probes: Mapping[str, Probe] | None) -> bool:
    if probes is not None and component in probes:
        return probes[component](root)
    if component == "python":
        return _python_ready(root)
    if component == "gpu":
        return _gpu_ready(root)
    if component == "model":
        return _model_ready()
    assert_never(component)


def _call_installer(
    component: ComponentId,
    root: Path,
    progress: Progress | None,
    installers: Mapping[str, Installer] | None,
) -> None:
    if installers is not None and component in installers:
        installers[component](root, progress or (lambda _message: None))
        return
    sink = progress or (lambda _message: None)
    if component == "python":
        _install_python(root, sink)
        return
    if component == "gpu":
        _install_gpu(root, sink)
        return
    if component == "model":
        _install_model(sink)
        return
    assert_never(component)


def _call_remover(component: ComponentId, root: Path, app: str, removers: Mapping[str, Remover] | None) -> None:
    if removers is not None:
        if component not in removers:
            raise SuiteInstallError(f"no remover for {component}")
        removers[component](root)
        return
    if component == "python":
        _remove_python(root)
        return
    if component == "gpu":
        _remove_gpu(root)
        return
    if component == "model":
        _remove_model(app)
        return
    assert_never(component)


def _register_model_user(app: str) -> None:
    try:
        register(app)
    except (OSError, StoreError):
        return


def _python_executable(root: Path) -> str | None:
    folder = root / "python"
    if sys.platform == "win32":
        candidate = folder / "python.exe"
    else:
        candidate = folder / "bin" / "python3"
    if candidate.is_file():
        return str(candidate)
    pointer = folder / "interpreter.json"
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    executable = data.get("executable")
    version = str(data.get("version") or "")
    if not isinstance(executable, str) or not Path(executable).is_file():
        return None
    if version != PYTHON_VERSION and not version.startswith(f"{PYTHON_VERSION}."):
        return None
    return executable


def _python_ready(root: Path) -> bool:
    executable = _python_executable(root)
    if executable is None:
        return False
    script = "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)"
    completed = subprocess.run([executable, "-c", script], check=False, capture_output=True)
    return completed.returncode == 0


def _install_python(root: Path, progress: Progress) -> None:
    dest = root / "python"
    bundled = bundled_python_tree()
    if bundled is not None:
        try:
            same = bundled.resolve() == dest.resolve()
        except OSError:
            same = False
        if not same:
            progress("Copying Python 3.14 into the shared suite folder")
            if dest.is_symlink():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
            try:
                shutil.copytree(bundled, dest, symlinks=True)
            except OSError as exc:
                raise SuiteInstallError(f"could not copy Python 3.14: {exc}") from exc
            if _python_ready(root):
                return
            if dest.is_symlink():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
    if sys.version_info[:2] == (3, 14):
        progress("Recording the current Python 3.14 interpreter")
        dest.mkdir(parents=True, exist_ok=True)
        payload = {"executable": sys.executable, "version": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"}
        (dest / "interpreter.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return
    raise SuiteInstallError("Python 3.14 was not found. Install the Enlarge bundle, which carries that runtime.")


def _remove_python(root: Path) -> None:
    dest = root / "python"
    if dest.is_symlink():
        dest.unlink()
        return
    if not dest.exists():
        return
    if root.resolve() not in dest.resolve().parents:
        raise SuiteInstallError("refusing to delete a Python outside the shared folder")
    shutil.rmtree(dest)


def _gpu_ready(root: Path) -> bool:
    python = _python_executable(root) or sys.executable
    if not Path(python).is_file():
        return False
    completed = subprocess.run([python, "-c", _GPU_PROBE], check=False, capture_output=True)
    return completed.returncode == 0


def _install_gpu(root: Path, progress: Progress) -> None:
    python = _python_executable(root) or sys.executable
    install_gpu_wheels(python, progress)


def _remove_gpu(root: Path) -> None:
    python = _python_executable(root) or sys.executable
    if python and Path(python).is_file() and not getattr(sys, "frozen", False):
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "torch",
                "onnxruntime",
                "onnxruntime-gpu",
                "tensorrt-cu13",
                "tensorrt",
                "tensorrt-cu13-bindings",
                "tensorrt-cu13-libs",
                "spandrel",
                "onnx",
            ],
            check=False,
        )


def _default_model() -> ModelManifest:
    catalog = load_builtin(release=True)
    model = catalog.get(DEFAULT_MODEL_ID)
    if model is None:
        raise SuiteInstallError("the default model is not in the catalogue")
    return model


def _model_ready() -> bool:
    model = _default_model()
    if cached_onnx(model) is None:
        return False
    if model.onnx_url and model.onnx_sha256:
        return True
    state, _detail = weight_report(model)
    return state == "ready"


def _install_model(progress: Progress) -> None:
    model = _default_model()

    def on_progress(_fraction: float, message: str) -> None:
        progress(message)

    try:
        ensure_onnx(model, on_progress=on_progress)
    except (StoreError, OnnxExportError) as exc:
        raise SuiteInstallError(str(exc)) from exc


def _remove_model(app: str) -> None:
    others = unregister(app)
    if others:
        return
    model = _default_model()
    if not model.filename or "/" in model.filename or "\\" in model.filename:
        raise SuiteInstallError(f"{model.id} has an unsafe filename")
    target = models_dir() / model.filename
    target.unlink(missing_ok=True)
    target.with_name(target.name + ".partial").unlink(missing_ok=True)
    cached = cached_onnx(model)
    if cached is not None:
        cached.unlink(missing_ok=True)
    if model.sha256:
        (models_dir() / f"{model.sha256.lower()}.onnx").unlink(missing_ok=True)


def _pip(python: str, args: list[str], message: str, progress: Progress | None, *, required: bool) -> None:
    _emit(progress, message)
    completed = subprocess.run([python, "-m", "pip", *args], check=False)
    if required and completed.returncode != 0:
        raise SuiteInstallError(f"{message} failed ({completed.returncode})")
