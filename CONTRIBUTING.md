# Contributing

VOXPrint AI Enlarge is the still-image upscaler in the VOXPrint AI Media Suite. Issues and pull requests are welcome.

## What belongs here

* Bug reports with the version from `--version`, the operating system, and the GPU driver.
* Small, focused pull requests. One change per request is easier to review.
* Documentation that matches the command the program actually accepts.

Model weights are not stored in this repository. A pull request must not add a `.pth` or `.onnx` weight file.

## Setup

Python 3.14:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
python -m ruff check enlarge voxprint_engine main.py tools tests
python -m mypy
```

The window tests need an offscreen display:

```bash
QT_QPA_PLATFORM=offscreen VOXPRINT_CI_LANCZOS=1 python -m pytest
```

`VOXPRINT_CI_LANCZOS=1` is for tests. It is not a user setting.

## Style

Public Python uses Google docstrings. Imports stay at the top of the file. A switch on a union or an enum ends with a `never` check so a new variant fails the type checker until it is handled.

The shared engine in `voxprint_engine` does not import Qt. `API` stays 1 until a breaking change. Adding a public name updates `voxprint_engine/api.py` and `tests/test_api.py` in the same pull request.

## Licence

This project is Apache-2.0. By submitting a contribution you agree that it can be distributed under that licence. Model manifests must use a licence the release gate accepts. Non-commercial weights are refused.
