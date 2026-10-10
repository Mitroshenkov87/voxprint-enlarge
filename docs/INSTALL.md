# Install and configuration

Phase 1 has no installer. Run it from a checkout.

## Requirements

* Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later
* An NVIDIA RTX 40 (Ada, compute capability 8.9) or newer for TensorRT. CUDA is the path when that engine does not build. Lanczos and the CPU ONNX smoke test do not need a GPU. GPU packages and a local benchmark are in `docs/ENGINE.md`.
* Python 3.11 or newer

## From source

```bash
python -m pip install -r requirements.txt
python main.py
```

Smoke test, no window left open:

```bash
python main.py --selftest
```

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

On Windows the shortcut belongs in the shared **Voxprint** folder, next to the other suite apps. `installer/windows/start-menu.ps1` creates that shortcut for a checkout. There is no silent installer switch yet.

On Linux, `installer/linux/voxprint-enlarge.desktop` is the desktop entry. The icon file is `installer/linux/voxprint-enlarge-256.png`.

## CLI

```text
python main.py                         open the window
python main.py --selftest              open the window, then quit
python main.py --version
python main.py --upscale FILE --out FILE [--scale 2|4] [--format avif|jxl|webp|jpeg|heic] [--quality N] [--lossless] [--model ID] [--double]
python main.py --worker                internal worker process
```
