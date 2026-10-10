# Voxprint AI Enlarge

Part of the Voxprint AI Media Suite, together with [Voxprint AI Audiobook Builder](https://github.com/Mitroshenkov87/voxprint-audiobook-builder) and Voxprint AI Movie Dubber.

Voxprint AI Enlarge makes a still picture larger on your own computer. You pick images or a folder, pick a model, choose the scale and the file format, and run. A before/after slider and a batch queue show the result. Nothing is uploaded.

Saved files default to AVIF. JPEG XL (including lossless) and WebP are the other main formats. JPEG is a legacy option. HEIC is a legacy option only when this computer already has a codec; the app does not bundle a HEIC encoder. PNG files can be opened. They are not an output format.

**Status:** pre-alpha. Run uses ONNX Runtime on TensorRT. CUDA is used when that engine does not build. Weights for the three pinned models download into the shared Voxprint models folder.

**Images only.** Video restoration belongs to Voxprint AI Clarity. Enlarge does not open video files.

**Requirements:** Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later. Python 3.14. Enlarge needs an NVIDIA GPU that CUDA can use, and an NVIDIA driver from the 600 series or newer. Without that GPU, or with an older driver, Enlarge does not start and shows a clear message. An NVIDIA GPU below compute capability 8.9 (RTX 40 / Ada) still starts, and the diagnostics warning is the only extra signal. The x86-64 CPU baseline is x86-64-v3, with no AVX-512 requirement, and the same code runs on ARM. GPU setup and a local benchmark are in `docs/ENGINE.md`.

**License:** Apache-2.0. See `LICENSE` and `NOTICE`. Model weights keep the licence recorded in the registry. CC-BY models are listed on the Credits screen. A release build refuses non-commercial or unlicensed weights.

## Run from source

```bash
python -m pip install -r requirements.txt
python main.py
```

```bash
python main.py --upscale samples/sample.png --out out.avif --scale 2
python -m pytest
```

Settings shared with the other Voxprint apps live in `suite.json`. Install paths, the Start menu folder name (`Voxprint`), and the CLI are in `docs/INSTALL.md`. The shared engine that Clarity imports is described in `docs/ARCHITECTURE.md` and `docs/ENGINE.md`.

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network.
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.
