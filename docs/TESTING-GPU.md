# GPU VM checklist

Use this on the RTX 4090 machine (Windows 11, Windows Server 2025, or Linux, NVIDIA driver from the 600 series or newer). CI does not run these steps. The installer contains no model weights.

Install `v0.2.0-rc.5` (codename Chayil). Check `SHA256SUMS` before you install. A stock machine can run the short set with `bash scripts/vm_test.sh --smoke` before anything else. The script prints `python: ...` for the interpreter it chose: the shared runtime (`%LOCALAPPDATA%\Voxprint\shared\python\python.exe`, or `~/.local/share/voxprint/shared/python/bin/python3`), then `py -3`, then `python3`. `python3` on Windows is often the Microsoft Store alias, so it is last. Exit 0 is pass, 1 is a failed test, 2 is setup, and 3 means no usable GPU.

* Windows: run `VoxprintEnlarge-Setup.exe`. It installs for the current user under `%LOCALAPPDATA%\Voxprint\Enlarge`.
* Linux: unpack `voxprint-enlarge-linux-x64.tar.gz` and run `./install.sh`. The command is then `voxprint-enlarge`.

The installer fills the shared suite folder itself: Python 3.14, the CUDA wheels, and the default model, converted to ONNX. Do not export a weight by hand, and do not install TensorRT yourself. A stock install must upscale.

## Quick re-test

Unset `VOXPRINT_HOME` when `--selftest` warns that it differs from the default. Then:

```bash
bash scripts/vm_test.sh --smoke
voxprint-enlarge upscale --input line.png --scale 4 --preset art --format avif --json-progress
rm -f "$LOCALAPPDATA/Voxprint/models/1f2be2b4786b5031776aed90e1818564f5576ab1e44da9355c7f3788ea25bdea.onnx"
voxprint-enlarge upscale --input photo.jpg --scale 4 --preset photo --quality-tier max --format avif --json-progress
"$LOCALAPPDATA/Voxprint/Enlarge/VoxprintEnlarge.exe"
```

`vm_test.sh --smoke` prints `python:` and covers the stock upscale, the cached rerun, anime 6B at 4×, and the long-side chain. The line-art command is the same anime 6B run on a real drawing, after Check hash on `realesrgan-x4plus-anime6b`. Removing that ONNX file forces the first DAT2 `.pth` conversion. The window title is `Voxprint AI Enlarge` and the client area is opaque.

Run the cases that failed on `v0.2.0-rc.2` first. The anime 6B run and the Canvas chain stay as regressions after those.

## 1. A stock install upscales

```bash
voxprint-enlarge --version
voxprint-enlarge --selftest --json
voxprint-enlarge --capabilities --json
voxprint-enlarge upscale --input samples/sample.png --scale 2 --format avif --json-progress
```

`--version` exits 0, prints `0.2.0-rc.5` and `Chayil`, and says `installed build`. It does not say `run from source`.

`--selftest --json` exits 0. `ok` is true. `tensorrt_ep.detail` and `cuda_ep.detail` are both `loaded`. `voxprint_home` is the resolved folder and `voxprint_home_source` is `default` on a stock install. A line on stderr that starts with `warning: VOXPRINT_HOME` means the variable points away from a default home that already has `shared/manifest.json`. Unset `VOXPRINT_HOME` and install again so the apps share one runtime.

`--capabilities --json` exits 0. On this machine `gpu.status` is `ok` and `gpu.reason` is null. `version` matches `--version`. The default model is `installed` true. It carries the same `voxprint_home` and `voxprint_home_source` fields.

The upscale exits 0 and writes an AVIF file next to the sample. Nobody has to produce an ONNX file first. The first run reports `building_engine`, and `eta_s` starts near 90. Run the same command again. It reports `loading_cached_engine` and finishes sooner.

## 2. A bad scale is one JSON line

```bash
voxprint-enlarge upscale --input samples/sample.png --scale 3 --json-progress
```

Exit 2. Stdout is one JSON object. `type` and `op` are `error`, and `code` is `bad_input`. The output is not argparse usage text.

## 3. Photo, max, on CUDA

`4x-nomos2-hq-dat2` failed inside TensorRT on `v0.2.0-rc.2` (`EP_FAIL` on `TRTKernel_graph_main_graph`, about 25 s, no file). This build runs that model on the CUDA provider. On the Model step the row says `CUDA (TensorRT cannot run this network)`.

Download `4x-nomos2-hq-dat2` and press Check hash. Use a noisy, compressed photograph, long side at least 2000 px:

```bash
voxprint-enlarge upscale --input photo.jpg --scale 4 --preset photo --quality-tier max --format avif --json-progress
```

The job writes an AVIF file. It does not stop on `EP_FAIL`. Record wall time, `done.ms`, and peak VRAM from `nvidia-smi`.

## 4. Regressions

These passed on the same 4090 against `v0.2.0-rc.2` once an ONNX file and TensorRT 10.14.1.48 were available. They must pass from this stock install with neither step done by hand.

**Line art.** An anime or line-art image, x4, RealESRGAN anime 6B. The previous run took 77 s, peak 1.6 GB, and wrote 3200×2152 AVIF from an 800×538 source (first run, including the TensorRT build, `done.ms` 72017).

```bash
voxprint-enlarge upscale --input line.png --scale 4 --preset art --format avif --json-progress
```

Download `realesrgan-x4plus-anime6b` from the Model step first, and press Check hash.

**Hand-off chain.** A roughly 2K image (the previous file was 2048×2061). Another process already holds `%TEMP%/voxprint-gpu.lock` (`$TMP/voxprint-gpu.lock` on Linux): JSON `owner`, `pid` of a process that is still alive, `job`, `started`, and `eta` about an hour ahead, ISO-8601 UTC. Pass that pid:

```bash
voxprint-enlarge upscale --input photo-2k.png --output out.avif --long-side 3840 --format avif --json-progress --parent-lock-pid <pid>
```

Preset and quality stay at their defaults (`photo`, `fast`) when you omit them. The long side of the file is 3840. The command does not open a window. The previous machine took 90 s for the first TensorRT build (`done.ms` 85695, result 3816×3840) and 23.7 s when that engine was cached (`done.ms` 18575). After `done`, the lock file is still there and its `pid` is still the caller's. A normal job, with no parent pid, removes the lock when it finishes.

## 5. What to record

For each GPU run:

* Wall time, and `ms` on the `done` line. Keep the first TensorRT build separate from the cached second run.
* Peak VRAM from `nvidia-smi` (`memory.used`) while tiles are running.
* Whether work spilled into system memory: `memory.used` flat while the run slows and retries a smaller tile, or shared GPU memory climbing in `nvidia-smi`.
* The lock: created and removed for a normal job; left untouched, same pid, for `--parent-lock-pid`.
* For the photo-max run, that the file exists and the log has no `EP_FAIL`.

`done` carries `output`, `width`, `height`, `format`, and `ms`. x4 of a W×H image is 4W×4H. The long-side job is 3840 on the long side.

## 6. Crops for review

From the photo-max run and the line-art run, save a few full-resolution crops of the same region, before and after. Label each file with the model and the scale. These are for a person to look at. They are not a score.
