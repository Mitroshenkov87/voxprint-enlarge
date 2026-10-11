"""Shared ``suite.json`` (schema 1).

``gpu`` and ``models_dir`` are read and written through
``voxprint_common.suite_settings``. Unknown keys stay in the file.
"""
from __future__ import annotations

from pathlib import Path

from voxprint_common.paths import models_dir as resolved_models_folder
from voxprint_common.suite_settings import file
from voxprint_common.suite_settings import gpu as shared_gpu
from voxprint_common.suite_settings import models_dir as shared_models_dir
from voxprint_common.suite_settings import set_gpu as store_gpu
from voxprint_common.suite_settings import set_models_dir as store_models_dir

SCHEMA = 1
FILENAME = "suite.json"
DEFAULT_GPU = "auto"


def path() -> Path:
    """Path of the shared suite.json."""
    return file()


def gpu() -> str:
    """The saved GPU choice, or auto when the file has none."""
    return shared_gpu()


def models_dir() -> Path | None:
    """Absolute folder from the file, or None when the key is missing, null, or relative."""
    return shared_models_dir()


def resolved_models_dir() -> Path:
    """``VOXPRINT_MODELS_DIR``, else the suite value, else the default shared folder."""
    return resolved_models_folder()


def set_gpu(value: str) -> str:
    """Store a GPU choice and leave every other suite key alone."""
    return store_gpu(value)


def set_models_dir(folder: Path | None) -> None:
    """None stores JSON null, which means the default shared folder."""
    store_models_dir(folder)
