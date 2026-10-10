# Voxprint AI Enlarge

Part of the Voxprint AI Media Suite, together with [Voxprint AI Audiobook Builder](https://github.com/Mitroshenkov87/voxprint-audiobook-builder) and Voxprint AI Movie Dubber.

Voxprint AI Enlarge makes a still picture larger on your own computer. You pick images or a folder, pick a model, choose the scale and the file format, and run. A before/after slider and a batch queue show the result. Nothing is uploaded.

**Images only.** Video restoration belongs to Voxprint AI Clarity. Enlarge does not open video files.

**Status:** `v0.1.0-rc.1` is the installable release candidate for a GPU machine. A source checkout still reports pre-alpha until a release build stamps it. Run uses ONNX Runtime on TensorRT. CUDA is used when that engine does not build. Weights for the three pinned models download into the shared Voxprint models folder. The installer and the tarball do not contain those weights.

**License:** Apache-2.0. See `LICENSE` and `NOTICE`. Model weights keep the licence recorded in the registry. A release build refuses non-commercial or unlicensed weights.

The VM checklist is [docs/TESTING-GPU.md](docs/TESTING-GPU.md). Later work is listed in [docs/ROADMAP.md](docs/ROADMAP.md) and is not in this build.

## System requirements

* Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later.
* An NVIDIA GPU that CUDA can use, and an NVIDIA driver from the 600 series or newer. An RTX 40 series card (compute capability 8.9) or newer is recommended. A card below that still starts and shows a diagnostics warning.
* There is no CPU mode. Without that GPU, or with an older driver, the window does not start and shows a clear message. `--capabilities --json` and `--serve --hidden` still start in that case, so another app can see there is nothing to run.
* The x86-64 CPU baseline is x86-64-v3. There is no AVX-512 requirement. The same code runs on ARM, and the GPU path still needs NVIDIA CUDA.
* A network connection the first time, for `--install-gpu` and for the model download. After that, a run is offline.

Check `SHA256SUMS` from the GitHub release against the file you downloaded before you install.

## Windows

1. Run `VoxprintEnlarge-Setup.exe`. It does not need an administrator. It installs for the current user under `%LOCALAPPDATA%\Voxprint\Enlarge`.
2. The Start menu folder **Voxprint** gains **Voxprint AI Enlarge**. The installer registers the Lucent hand-off. Uninstall removes that registration before it deletes the files.
3. Open **Voxprint AI Enlarge** from that Start menu folder. The first window is the Images step.
4. Once, from a terminal, install the GPU wheels into the bundled Python (this needs the network, and it does not download models):

   ```text
   "%LOCALAPPDATA%\Voxprint\Enlarge\VoxprintEnlarge.exe" --install-gpu
   ```

5. On **Model**, select a pinned model, press **Download**, and wait for the progress line. **Check hash** confirms the sha256. **Remove** deletes the file only when no other Voxprint app is listed in the shared `.users.json`.
6. On **Images**, press **Add images** or **Add folder**. **Include subfolders** is on by default. On **Output**, leave the format at AVIF and pick **x2** or **x4**. On **Run**, press **Run**.
7. The first TensorRT engine for that model and GPU is built during the run. The status line says it is building the engine. The cache is `%LOCALAPPDATA%\Voxprint\trt-cache`. The next run of the same model loads the cached engine and is shorter.
8. A folder batch shows one row per image, with its own progress. A failed image is recorded and the queue continues. **Cancel** stops the image that is running. If the app closes while a row is still unfinished, the next launch restores the queue and **Run** continues those images.
9. Open a finished row on **Run**. The divider slides between the original and the result. The scroll wheel zooms both pictures together, and dragging pans both.

The window writes `<name>_x<scale>.avif` next to the source. It does not overwrite an existing file.

## Linux

1. Unpack the release tarball and run the installer from inside it:

   ```bash
   tar -xzf voxprint-enlarge-linux-x64.tar.gz
   cd voxprint-enlarge
   ./install.sh
   ```

2. `install.sh` copies the tree to `~/.local/share/voxprint/enlarge`. It puts `voxprint-enlarge` on `PATH` via `~/.local/bin` (that directory needs to be on your `PATH`). It also writes two desktop files under `~/.local/share/applications/`:

   * `voxprint-enlarge.desktop` is the launcher you see in the application menu.
   * `ai.voxprint.Enlarge.desktop` is the hidden discovery file for the Lucent hand-off.

3. Start **Voxprint AI Enlarge** from the application menu, or run `voxprint-enlarge`.
4. Once, install the GPU wheels:

   ```bash
   voxprint-enlarge --install-gpu
   ```

5. Download a model, run a scale, and use the batch and the compare view the same way as on Windows. The TensorRT cache is `~/.cache/voxprint/trt-cache`.

`uninstall.sh` in the installed tree unregisters the hand-off and removes the install directory.

## Formats

Saved files default to AVIF. JPEG XL, including lossless JPEG XL, and WebP are the other main formats. JPEG is labelled legacy and is not the default. HEIC is a legacy option only when this computer already has a codec. The app does not bundle a HEIC encoder. PNG files can be opened. They are not an output format.

## Model credits

The Credits button builds this list from the registry. The three models this build can run:

| Model | Author | Licence |
|---|---|---|
| 4xNomos2_realplksr_dysample (photo, fast) | Philip Hofmann | CC-BY-4.0 |
| 4xNomos2_hq_dat2 (photo, max) | Philip Hofmann | CC-BY-4.0 |
| RealESRGAN x4plus anime 6B (art) | Xintao Wang and contributors | BSD-3-Clause |

CC-BY models need that credit when the weights are used. RealESRGAN is BSD-3-Clause. Other registry rows are recorded for a later phase and are not selectable in this build.

## Run from source

Python 3.14:

```bash
python -m pip install -r requirements.txt
python main.py
```

```bash
python main.py --upscale samples/sample.png --out out.avif --scale 2
python -m pytest
```

Settings shared with the other Voxprint apps live in `suite.json`. Install paths, the Start menu folder name (`Voxprint`), and the CLI are in `docs/INSTALL.md`. The shared engine that Clarity imports is described in `docs/ARCHITECTURE.md` and `docs/ENGINE.md`. The protocol another Voxprint app uses to hand an image to Enlarge is `docs/IPC.md`.

## Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee.

* **Network:** TRON (TRC-20)
* **Accepted:** USDT or TRX
* **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

Send funds only via the TRON (TRC-20) network. Sending assets from another network to this address loses them.
