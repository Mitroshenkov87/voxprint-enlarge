# GPU VM checklist

Use this on a rented Windows 11 or Linux machine with an RTX 4090 (or another NVIDIA GPU) and a driver from the 600 series or newer. The GitHub build does not run these steps. CI has no GPU, and the installer does not contain model weights.

Install the `v0.1.0-rc.1` pre-release first. The steps below match the README.

* Windows: run `VoxprintEnlarge-Setup.exe`. It installs for the current user under `%LOCALAPPDATA%\Voxprint\Enlarge`, adds **Voxprint AI Enlarge** to the Start menu folder **Voxprint**, and writes the Lucent discovery key. Uninstall runs `--unregister` before it deletes the files.
* Linux: unpack `voxprint-enlarge-linux-x64.tar.gz` and run `./install.sh`. That copies the tree to `~/.local/share/voxprint/enlarge`, puts `voxprint-enlarge` on `PATH`, and writes the discovery desktop file.

Check `SHA256SUMS` against the downloaded files before installing.

## 1. Driver

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
voxprint-enlarge --capabilities --json
```

`nvidia-smi` shows the 4090 and a driver major of 600 or newer. The JSON exits 0. `gpu.status` is `ok` for an RTX 4090 on that driver. `gpu.status` `none` with `driver_too_old` means the driver is below 600. `no_nvidia` means CUDA cannot see a GPU. A card below compute capability 8.9 would be `warn` and would still run.

## 2. GPU wheels, once

The bundle ships the CPU ONNX Runtime wheel so the app can start. TensorRT is installed on this machine:

```bash
voxprint-enlarge --install-gpu
```

That removes `onnxruntime` and installs `onnxruntime-gpu` and `tensorrt-cu13` (CUDA 13). It needs a network. It does not download model weights.

## 3. First TensorRT build

Open the app from the Start menu or with `voxprint-enlarge`. Add `samples/sample.png` (or any small PNG). Choose the fast photo model **4xNomos2 RealPLKSR DySample**. Run scale x2.

The status line says `building the TensorRT engine` the first time. The engine file lands under the TensorRT cache (`%LOCALAPPDATA%\Voxprint\trt-cache` or `~/.cache/voxprint/trt-cache`). Run the same image again. The status line says `loading the cached TensorRT engine`, and the run is shorter.

## 4. Three models at x2 and x4

Download each pinned model from the Model step (progress, then Check hash). Then run one still image at x2 and at x4:

| Model | Preset |
|---|---|
| 4xNomos2 RealPLKSR DySample | photo, fast |
| 4xNomos2 HQ DAT2 | photo, max |
| RealESRGAN x4plus anime 6B | art |

Record for each of the six runs:

* output path, width, and height (x2 of a W×H image is 2W×2H, x4 is 4W×4H)
* wall time
* peak VRAM from `nvidia-smi` while the tiles are running (`memory.used`)

The output sits next to the source as `<name>.enlarged-<scale>x.avif` when the job comes from Lucent, and as `<name>_x<scale>.avif` from the window. An existing file is not overwritten.

Delete one downloaded weight from the Model step after the runs. With only Enlarge listed in `<models>/.users.json`, the file is removed. If another suite app is still true in that file, the delete is refused.

## 5. Batch, compare, cancel

Add a folder that contains a subfolder of images. Include subfolders stays on. Run the queue. Each row shows its own progress. Cancel during a later image. The finished files stay. Quit the app while a row is still running, then open it again. The queue comes back and Run continues the images that are not done.

Open a finished row. The divider slides, and the scroll wheel zooms both pictures together.

Credits lists the registry, including the CC-BY Nomos lines and the BSD RealESRGAN lines.

## 6. Lucent hand-off

With the app not already serving:

```bash
voxprint-enlarge --serve --hidden
```

From another terminal, a one-shot client can use the CLI fallback, which speaks the same JSON lines:

```bash
voxprint-enlarge --input samples/sample.png --scale 2 --preset photo --quality fast --json-progress
```

Exit 0, and the last line is `done` with an `output` path. A socket client (see `docs/IPC.md`) sends `hello`, then `enlarge.request`, and reads `accepted`, `progress`, and `done`. `open_ui` true opens the window with that image loaded.

`--capabilities --json` stays exit 0 even if you later hide the GPU. The window itself still does not open without a usable NVIDIA GPU.
