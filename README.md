<p align="center">
  <img src="assets/enlarge-256.png" alt="Voxprint AI Enlarge" width="96">
</p>

<h1 align="center">Voxprint AI Enlarge</h1>

<p align="center">Offline image upscaling for the Voxprint suite. Nothing is uploaded.</p>

<p align="center">
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <a href="https://github.com/Mitroshenkov87/voxprint-enlarge/releases"><img alt="Releases" src="https://img.shields.io/github/v/release/Mitroshenkov87/voxprint-enlarge?include_prereleases&label=release"></a>
  <a href="https://www.python.org/"><img alt="Python 3.14" src="https://img.shields.io/badge/python-3.14-blue"></a>
  <a href="docs/CLI.md"><img alt="Command line" src="https://img.shields.io/badge/docs-command%20line-lightgrey"></a>
</p>

Part of the Voxprint AI Media Suite, together with [Voxprint AI Audiobook Builder](https://github.com/Mitroshenkov87/voxprint-audiobook-builder) and Voxprint AI Movie Dubber.

Enlarge makes a still picture larger on your own computer. You pick images or a folder, pick a model, choose the scale and the file format, and run. A before/after slider and a batch queue show the result.

**Images only.** Video restoration belongs to Voxprint AI Clarity. Enlarge does not open video files.

The docs site is [mitroshenkov87.github.io/voxprint-enlarge](https://mitroshenkov87.github.io/voxprint-enlarge/).

## Screenshots

<p align="center">
  <img src="docs/images/enlarge-main.webp" alt="The Images step of Voxprint AI Enlarge" width="840">
</p>

Illustrated picture, RealESRGAN anime 6B at 4×:

<p align="center">
  <img src="docs/images/anime-before.webp" alt="Before, illustrated crop" width="360">
  <img src="docs/images/anime-after.webp" alt="After, 4× RealESRGAN anime 6B" width="360">
</p>

Photograph, long side from 2048 to 3840:

<p align="center">
  <img src="docs/images/photo-before.webp" alt="Before, photo crop" width="360">
  <img src="docs/images/photo-after.webp" alt="After, long side 3840" width="360">
</p>

## Features

* Still images only. A folder batch keeps one row per picture, and an unfinished queue is restored the next time the window opens.
* Three pinned models: a fast photo model, a higher-quality photo model, and an illustrated model. Weights download into the shared Voxprint models folder. The installer does not contain those weights.
* ONNX Runtime on TensorRT, with the CUDA provider when TensorRT cannot run that network. The first engine build is cached and the next run of the same model is shorter.
* AVIF by default. JPEG XL, including lossless JPEG XL, and WebP are the other main formats. JPEG is labelled legacy. HEIC is available only when this computer already has a codec. PNG can be opened and is not an output format.
* A before/after divider. The scroll wheel zooms both pictures together, and dragging pans both.
* A headless command other suite apps can call. The contract is [docs/CLI.md](docs/CLI.md).

## Requirements

* Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later.
* An NVIDIA GPU that CUDA can use, and an NVIDIA driver from the 600 series or newer. An RTX 40 series card (compute capability 8.9) or newer is recommended. A card below that still starts and shows a diagnostics warning.
* There is no CPU mode. Without that GPU, or with an older driver, the window does not start and shows a clear message. `--capabilities --json` and `--serve --hidden` still start in that case, so another app can see there is nothing to run.
* The x86-64 CPU baseline is x86-64-v3. There is no AVX-512 requirement. The same code runs on ARM, and the GPU path still needs NVIDIA CUDA.
* A network connection while the installer fills the shared suite folder (Python 3.14, the CUDA wheels, and the default model). A later launch repairs anything missing. After that, a run is offline.

**License:** Apache-2.0. See `LICENSE` and `NOTICE`. Model weights keep the licence recorded in the registry. A release build refuses non-commercial or unlicensed weights.

Check `SHA256SUMS` from the GitHub release against the file you downloaded before you install.

## Install

### Windows

1. Run `VoxprintEnlarge-Setup.exe`. It does not need an administrator. It installs for the current user under `%LOCALAPPDATA%\Voxprint\Enlarge`.
2. The Start menu folder **Voxprint** gains **Voxprint AI Enlarge**. The installer registers the Lucent hand-off. Uninstall removes that registration before it deletes the files.
3. The installer shows progress while it copies Python 3.14 to `%LOCALAPPDATA%\Voxprint\shared`, installs the CUDA wheels there, and downloads the default model. Enlarge and Clarity share that folder. To re-check it later:

   ```text
   "%LOCALAPPDATA%\Voxprint\Enlarge\VoxprintEnlarge.exe" --selftest
   ```

### Linux

1. Unpack the release tarball and run the installer from inside it:

   ```bash
   tar -xzf voxprint-enlarge-linux-x64.tar.gz
   cd voxprint-enlarge
   ./install.sh
   ```

2. `install.sh` copies the tree to `~/.local/share/voxprint/enlarge`. It puts `voxprint-enlarge` on `PATH` via `~/.local/bin` (that directory needs to be on your `PATH`). It also writes two desktop files under `~/.local/share/applications/`:

   * `voxprint-enlarge.desktop` is the launcher you see in the application menu.
   * `ai.voxprint.Enlarge.desktop` is the hidden discovery file for the Lucent hand-off.

3. `install.sh` shows progress while it copies Python 3.14 to `~/.local/share/voxprint/shared`, installs the CUDA wheels there, and downloads the default model. The launcher uses that shared Python when it is present. To re-check it later, run `voxprint-enlarge --selftest`.

`uninstall.sh` drops this app's hold on the shared folder, deletes a shared file only when no other suite app still uses it, unregisters the hand-off, and removes the install directory.

## Quick start

1. Open **Voxprint AI Enlarge**. The first window is the Images step.
2. On **Model**, the default model is already there after a finished install. Other pinned models still use **Download**. **Check hash** confirms the sha256. **Remove** deletes a file only when no other Voxprint app is listed in the shared `.users.json`. The highest-quality photo model runs on CUDA, because TensorRT cannot execute that network.
3. On **Images**, press **Add images** or **Add folder**. **Include subfolders** is on by default.
4. On **Output**, leave the format at AVIF and pick **x2** or **x4**.
5. On **Run**, press **Run**. The status line names the stage, including the first TensorRT engine build. The cache is `%LOCALAPPDATA%\Voxprint\trt-cache` on Windows and `~/.cache/voxprint/trt-cache` on Linux.
6. A folder batch shows one row per image. A failed image is recorded and the queue continues. **Cancel** stops the image that is running.
7. Open a finished row on **Run**. The divider slides between the original and the result.

The window writes `<name>_x<scale>.avif` next to the source. It does not overwrite an existing file.

One picture from a terminal:

```bash
voxprint-enlarge --input photo.png --scale 4 --json-progress
```

## Command line

The headless command other suite apps call is [docs/CLI.md](docs/CLI.md). A job prints one JSON object per line and exits with a code from that contract.

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

## Roadmap

The next version brings the command line up to everything the window does, chooses a model from the graphics memory the card can spare, downloads models from more than one mirror, and uses graphics memory, system memory, and disk so bigger models fit on 16 GB and 24 GB cards. Later versions add more upscaling models, including an optional face-restore model. The list is [ROADMAP.md](ROADMAP.md).

## Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee.

* **Network:** TRON (TRC-20)
* **Accepted:** USDT or TRX
* **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

Send funds only via the TRON (TRC-20) network. Sending assets from another network to this address loses them.
