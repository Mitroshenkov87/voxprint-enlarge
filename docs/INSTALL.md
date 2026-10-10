# Install and configuration

A checkout runs from source. A release build is a per-user Windows installer or a Linux tarball. Neither one contains model weights. On the machine with the NVIDIA driver, run `voxprint-enlarge --install-gpu` once before the first TensorRT build. The VM checklist is `docs/TESTING-GPU.md`.

## Requirements

* Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later
* An NVIDIA GPU that CUDA can use, with a driver from the 600 series or newer. Without that GPU, or with an older driver, the window does not open. `--capabilities --json` and `--serve --hidden` still start, and a job then reports that the GPU is missing. An NVIDIA GPU below compute capability 8.9 (RTX 40 / Ada) still starts and shows a diagnostics warning. TensorRT is the session. CUDA is the path when that engine does not build. GPU packages and a local benchmark are in `docs/ENGINE.md`. The hand-off protocol is `docs/IPC.md`.
* Python 3.14

## From source

```bash
python -m pip install -r requirements.txt
python main.py
```

Smoke test, no window left open:

```bash
python main.py --selftest
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
| TensorRT engine cache (rebuildable, not refcounted) | `%LOCALAPPDATA%\Voxprint\trt-cache\<gpu>-<driver>-<trt>` or `~/.cache/voxprint/trt-cache/...` |
| Logs | `<Enlarge home>/logs/enlarge.log` |

`VOXPRINT_HOME`, `VOXPRINT_ENLARGE_HOME`, `VOXPRINT_MODELS_DIR`, `VOXPRINT_TRT_CACHE`, and `VOXPRINT_GPU_LOCK` override those locations. Enlarge does not change `ui_language` or `theme` in `suite.json`.

## Start menu

On Windows the shortcut belongs in the shared **Voxprint** folder, next to the other suite apps. `installer/windows/enlarge.iss` is the per-user installer. `installer/windows/start-menu.ps1` creates that shortcut for a checkout.

On Linux, `installer/linux/voxprint-enlarge.desktop` is the desktop entry. The icon file is `installer/linux/voxprint-enlarge-256.png`.

## CLI

```text
python main.py                         open the window
python main.py --selftest              open the window, then quit
python main.py --version
python main.py --capabilities --json   one JSON object, exit 0 with or without a GPU
python main.py --serve --hidden        protocol 1 listener; the window stays closed until open_ui
python main.py --register [--exe PATH]   discovery records for Lucent
python main.py --unregister              remove those records
python main.py --install-gpu             CUDA 13 ONNX Runtime and TensorRT, on a GPU machine
python main.py --input FILE --scale 2|4 --preset photo|art [--model ID] [--quality fast|max] [--format FMT] [--output PATH] --json-progress
python main.py --upscale FILE --out FILE [--scale 2|4] [--format avif|jxl|webp|jpeg|heic] [--quality N] [--lossless] [--model ID] [--double]
python main.py --worker                internal worker process
```

`--json-progress` streams the same JSON objects as the socket. Exit 0 after `done`, 1 after `error`. The message list, discovery keys, and output names are in `docs/IPC.md`.
