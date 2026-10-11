# Install and configuration

A checkout runs from source. A release build is a per-user Windows installer or a Linux tarball. Neither one contains model weights. The installer runs `--install-suite` before it finishes: it copies Python 3.14 into the shared suite folder, installs the CUDA wheels, and downloads the default model, with progress on the console. A later launch repairs anything missing. `--selftest` runs that check without opening a window. The VM checklist for the published `v0.2.0-rc.2` build, which still expects a manual `--install-gpu`, is `docs/TESTING-GPU.md`.

## Requirements

* Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later
* An NVIDIA GPU that CUDA can use, with a driver from the 600 series or newer. Without that GPU, or with an older driver, the window does not open. `--capabilities --json` and `--serve --hidden` still start, and a job then reports that the GPU is missing. An NVIDIA GPU below compute capability 8.9 (RTX 40 / Ada) still starts and shows a diagnostics warning. TensorRT is the session. CUDA is the path when that engine does not build. GPU packages and a local benchmark are in `docs/ENGINE.md`. The hand-off protocol is `docs/IPC.md`.
* Python 3.14

## From source

```bash
python -m pip install -r requirements.txt
python main.py
```

Re-check the shared install, no window:

```bash
python main.py --selftest
python main.py --selftest --json
```

Add a folder on the Images step to include subfolders. Run keeps a queue file and continues unfinished images after a restart. Cancel stops the image that is running. The Model step can download a pinned file, check its sha256, and remove it when no other Voxprint app is listed in `.users.json`. Credits are built from the registry.

One image, without the window:

```bash
python main.py --upscale samples/sample.png --out out.avif --scale 2 --format avif
```

## Where settings live

| What | Where |
|---|---|
| Shared suite settings (GPU, models folder, language, theme) | `%LOCALAPPDATA%\Voxprint\state\suite.json` on Windows, `~/.local/share/voxprint/state/suite.json` on Linux |
| Enlarge-only settings (scale, format, tile, output folder, custom models folder) | `%LOCALAPPDATA%\VoxprintEnlarge\settings.json` or `~/.local/share/voxprint-enlarge/settings.json` |
| Shared model weights | `%LOCALAPPDATA%\Voxprint\models` or `~/.local/share/voxprint/models`, with `.users.json` |
| Shared runtime (Python 3.14, CUDA wheels, refcounted manifest) | `%LOCALAPPDATA%\Voxprint\shared` or `~/.local/share/voxprint/shared` |
| TensorRT engine cache (rebuildable, not refcounted) | `%LOCALAPPDATA%\Voxprint\trt-cache\<gpu>-<driver>-<trt>` or `~/.cache/voxprint/trt-cache/...` |
| Logs | `<Enlarge home>/logs/enlarge.log` |

`VOXPRINT_HOME`, `VOXPRINT_ENLARGE_HOME`, `VOXPRINT_MODELS_DIR`, `VOXPRINT_TRT_CACHE`, and `VOXPRINT_GPU_LOCK` override those locations. Enlarge does not change `ui_language` or `theme` in `suite.json`.

## Start menu

On Windows the shortcut belongs in the shared **Voxprint** folder, next to the other suite apps. `installer/windows/enlarge.iss` is the per-user installer. `installer/windows/start-menu.ps1` creates that shortcut for a checkout.

On Linux, `installer/linux/voxprint-enlarge.desktop` is the desktop entry. The icon file is `installer/linux/voxprint-enlarge-256.png`.

## CLI

```text
python main.py                         open the window
python main.py --selftest [--json]    re-check the shared install and repair it
python main.py --version
python main.py --capabilities --json   one JSON object, exit 0 with or without a GPU
python main.py --serve --hidden        protocol 1 listener; the window stays closed until open_ui
python main.py --register [--exe PATH]   discovery records for Lucent
python main.py --unregister              remove those records
python main.py --install-suite           shared Python, CUDA wheels, and the default model
python main.py --uninstall-suite         drop this app's hold; delete a file only at zero holders
python main.py --install-gpu             torch, CUDA 13 ONNX Runtime, and TensorRT
python main.py --input FILE --scale 2|4 --preset photo|art [--model ID] [--quality fast|max] [--format FMT] [--output PATH] --json-progress
python main.py --upscale FILE --out FILE [--scale 2|4] [--format avif|jxl|webp|jpeg|heic] [--quality N] [--lossless] [--model ID] [--double]
python main.py --worker                internal worker process
```

The headless command other apps call is `docs/CLI.md`. `--json-progress` streams JSON lines on stdout and does not open a window. The socket messages are in `docs/IPC.md`. `python main.py --upscale FILE --out FILE` is unchanged.
