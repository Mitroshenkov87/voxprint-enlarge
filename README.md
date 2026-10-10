# Voxprint AI Enlarge

Part of the Voxprint AI Media Suite, together with [Voxprint AI Audiobook Builder](https://github.com/Mitroshenkov87/voxprint-audiobook-builder) and Voxprint AI Movie Dubber.

Voxprint AI Enlarge makes a still picture larger on your own computer. You pick images or a folder, pick a model, choose the scale and the file format, and run. A before/after slider and a batch queue show the result. Nothing is uploaded.

**Status:** pre-alpha. This build is the app skeleton. Run resizes with a Lanczos preview so the flow can be tried. Neural weights are not downloaded and not applied yet. The model you pick is remembered and sent to the worker for the later engine.

**Images only.** Video restoration belongs to Voxprint AI Clarity. Enlarge does not open video files.

**Requirements:** Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later, and an NVIDIA GPU for the neural engine. The preview resize in this build runs on the CPU.

**License:** Apache-2.0. See `LICENSE` and `NOTICE`. Model weights, when a later build downloads them, keep the licence recorded in the registry. CC-BY models are listed on the Credits screen.

## Run from source

```bash
python -m pip install -r requirements.txt
python main.py
```

```bash
python main.py --upscale samples/sample.png --out out.png --scale 2
python -m pytest
```

Settings shared with the other Voxprint apps live in `suite.json`. Install paths, the Start menu folder name (`Voxprint`), and the CLI are in `docs/INSTALL.md`. The shared engine that Clarity imports is described in `docs/ARCHITECTURE.md`.

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network.
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.
