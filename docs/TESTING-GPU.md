# GPU VM checklist

Use this on the RTX 4090 machine (Windows 11 or Linux, NVIDIA driver from the 600 series or newer). CI does not run these steps. The installer contains no model weights.

Install `v0.2.0-rc.2`. Check `SHA256SUMS` before you install.

* Windows: run `VoxprintEnlarge-Setup.exe`. It installs for the current user under `%LOCALAPPDATA%\Voxprint\Enlarge`.
* Linux: unpack `voxprint-enlarge-linux-x64.tar.gz` and run `./install.sh`. The command is then `voxprint-enlarge`.

Once, before the GPU cases, install the CUDA wheels (network, no model download):

```bash
voxprint-enlarge --install-gpu
```

This checklist matches the published v0.2.0-rc.2 installer, which does not fill the shared suite folder itself. Builds after that release run `--install-suite` from the installer (Python 3.14, the CUDA wheels, and the default model) into `%LOCALAPPDATA%\Voxprint\shared` or `~/.local/share/voxprint/shared`.

## 1. No-GPU checks first

Run these before any model download. They must not build a TensorRT engine.

```bash
voxprint-enlarge --version
voxprint-enlarge --capabilities --json
```

`--version` exits 0 and prints `0.2.0-rc.2` and `Bahir`. `--capabilities --json` exits 0. On this machine `gpu.status` is `ok`, `gpu.reason` is null, and `version` matches `--version`. `installed` is false until the weight file is on disk.

Dry-run and a bad argument, with `--json-progress`. Stdout is one JSON object per line. Every object has `"protocol": 1` and the same string in `type` and `op`.

```bash
voxprint-enlarge upscale --input samples/sample.png --scale 4 --dry-run --json-progress
voxprint-enlarge upscale --input samples/sample.png --scale 3 --json-progress
```

With the weights absent, the dry-run exits 7. The last line is `error` with `code` `model_missing` and `retryable` true. It creates no file. `--scale 3` exits 2 (`bad_input`). Other codes, if you hit them: `gpu_unsupported` 4, `gpu_busy` 5 (`retryable` true), `oom` 6, `io_error` 8, `cancelled` 130.

After the two models below are downloaded, repeat the dry-run. It exits 0: `accepted`, then `done` with the planned `output`, `width`, `height`, `format`, and `ms` 0. The file is not created.

## 2. Three GPU cases

Download only these two models from the Model step and press Check hash:

| Model | Id | Preset |
|---|---|---|
| 4xNomos2 HQ DAT2 | `4x-nomos2-hq-dat2` | photo, max |
| RealESRGAN x4plus anime 6B | `realesrgan-x4plus-anime6b` | art |

**Photo, tiling and VRAM.** A noisy, compressed photograph, long side at least 2000 px. Scale x4 with Nomos2 DAT2:

```bash
voxprint-enlarge upscale --input photo.jpg --scale 4 --preset photo --quality-tier max --format avif --json-progress
```

The first run of this model builds the TensorRT engine. Time that run with a clock. The status line says it is building the engine. The cache is `%LOCALAPPDATA%\Voxprint\trt-cache` or `~/.cache/voxprint/trt-cache`. Run the same command again. The status line says it is loading the cached engine, and the run is shorter. Record both times.

**Line art.** An anime or line-art image, x4, RealESRGAN anime 6B:

```bash
voxprint-enlarge upscale --input line.png --scale 4 --preset art --format avif --json-progress
```

**Hand-off chain.** A roughly 2K image (long side around 1920–2560). Another process already holds `%TEMP%/voxprint-gpu.lock` (`$TMP/voxprint-gpu.lock` on Linux): JSON `owner`, `pid` of a process that is still alive, `job`, `started`, and `eta` about an hour ahead, ISO-8601 UTC. Pass that pid:

```bash
voxprint-enlarge upscale --input photo-2k.png --output out.avif --long-side 3840 --format avif --json-progress --parent-lock-pid <pid>
```

Preset and quality tier stay at their defaults (`photo`, `fast`) when you omit them. The long side of the file is 3840. The command does not open a window. After `done`, the lock file is still there and its `pid` is still the caller's. A normal job, with no parent pid, removes the lock when it finishes.

## 3. What to record

For each GPU run:

* Wall time, and `ms` on the `done` line. Keep the first TensorRT build separate from the cached second run.
* Peak VRAM from `nvidia-smi` (`memory.used`) while tiles are running.
* Whether work spilled into system memory: `memory.used` flat while the run slows and retries a smaller tile, or shared GPU memory climbing in `nvidia-smi`.
* The lock: created and removed for a normal job; left untouched, same pid, for `--parent-lock-pid`.

`done` carries `output`, `width`, `height`, `format`, and `ms`. x4 of a W×H image is 4W×4H. The long-side job is 3840 on the long side.

## 4. Crops for review

From the photo run and the line-art run, save a few full-resolution crops of the same region, before and after. Label each file with the model and the scale. These are for a person to look at. They are not a score.
