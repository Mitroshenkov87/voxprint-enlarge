# Install and configuration

Phase 1 has no installer. Run it from a checkout.

## Requirements

* Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later
* An NVIDIA GPU for the neural engine in a later phase. This build's preview resize runs on the CPU.
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
python main.py --upscale samples/sample.png --out out.png --scale 2 --format png
```

## Where settings live

| What | Where |
|---|---|
| Shared suite settings (GPU, models folder, language, theme) | `%LOCALAPPDATA%\Voxprint\state\suite.json` on Windows, `~/.local/share/voxprint/state/suite.json` on Linux |
| Enlarge-only settings (scale, format, tile, output folder, custom models folder) | `%LOCALAPPDATA%\VoxprintEnlarge\settings.json` or `~/.local/share/voxprint-enlarge/settings.json` |
| Shared model weights (later phases) | `%LOCALAPPDATA%\Voxprint\models` or `~/.local/share/voxprint/models` |
| Logs | `<Enlarge home>/logs/enlarge.log` |

`VOXPRINT_HOME`, `VOXPRINT_ENLARGE_HOME`, `VOXPRINT_MODELS_DIR`, and `VOXPRINT_GPU_LOCK` override those locations. Enlarge does not change `ui_language` or `theme` in `suite.json`.

## Start menu

On Windows the shortcut belongs in the shared **Voxprint** folder, next to the other suite apps. `installer/windows/start-menu.ps1` creates that shortcut for a checkout. There is no silent installer switch yet.

On Linux, `installer/linux/voxprint-enlarge.desktop` is the desktop entry. The icon file is `installer/linux/voxprint-enlarge-256.png`.

## CLI

```text
python main.py                         open the window
python main.py --selftest              open the window, then quit
python main.py --version
python main.py --upscale FILE --out FILE [--scale 2|4] [--format png|jpeg|webp] [--quality N] [--model ID] [--double]
python main.py --worker                internal worker process
```
