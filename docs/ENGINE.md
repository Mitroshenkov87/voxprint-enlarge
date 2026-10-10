# Shared inference engine

`voxprint_engine` is the image upscaler Voxprint AI Enlarge owns and Voxprint AI
Clarity imports. It does not import Qt. `API` is 1.

```python
from voxprint_engine import API
from voxprint_engine.registry import combined_catalog, validate_manifest, LANCZOS_MODEL_ID
from voxprint_engine.worker import WorkerClient, UpscaleRequest
from voxprint_engine.infer import upscale_file, uses_lanczos
from voxprint_engine.store import ensure_weight, register, unregister
from voxprint_engine.paths import models_dir, trt_cache_dir
from voxprint_engine.vram import vram_budget_gb, choose_tile, halve_tile, note_spill
from voxprint_engine.gpu_lock import gpu_job
```

Clarity should keep using the worker for a long job. `upscale_file` is the
same function the worker calls, for a benchmark or a one-off frame.

## What Run does

The worker is `python -m voxprint_engine.worker`. One JSON object per line.
`engine` on an upscale request is `auto` (the default) or `onnx`.

Enlarge starts only when `nvidia-smi` reports an NVIDIA GPU whose driver major
version is 600 or newer. A computer without that GPU, or with an older driver,
does not start. The window shows the reason, and `--upscale` prints it and
exits. A card below compute capability 8.9 (RTX 40 / Ada) shows a diagnostics
warning and still starts.

`VOXPRINT_CI_LANCZOS=1` skips that gate and resamples the image with Lanczos.
Tests and CI set the flag so they stay offline and GPU-free. It is not a
setting in the app. A user job never takes that path.

`VOXPRINT_ORT_DEVICE=cpu`, `device` `cpu-onnx`, or `engine` `onnx` runs the
CPU execution provider. That is the smoke test. It is not a user mode.

Otherwise the session is:

1. TensorRT EP, fp16, engine cache enabled
2. CUDA EP, only when that TensorRT engine does not build (or when the TensorRT provider is not installed and CUDA is)

DirectML and Windows ML are not providers. The x86-64 CPU baseline is
x86-64-v3, without an AVX-512 path. ARM uses this same code.

If neither NVIDIA provider loads the file, and a `.pth` is present, the worker
loads it with PyTorch through `spandrel` (fp16 on CUDA, fp32 otherwise). If
`spandrel_extra_arches` is already imported, that fallback refuses to run.

The CPU execution provider is not in the GPU list. CI installs the CPU wheel
and sets `VOXPRINT_CI_LANCZOS=1`, so the tests do not download weights.

Downloads happen before `%TEMP%/voxprint-gpu.lock` is taken. The lock is held
for the whole inference, including the session build. The owner is `enlarge`.

## Models

Three weights are pinned. The sha256 is the official GitHub release asset.

| id | file | licence | role |
|---|---|---|---|
| `4x-nomos2-realplksr-dysample` | `4xNomos2_realplksr_dysample.pth` | CC-BY-4.0 | default, fast photos |
| `4x-nomos2-hq-dat2` | `4xNomos2_hq_dat2.pth` | CC-BY-4.0 | highest quality photos |
| `realesrgan-x4plus-anime6b` | `RealESRGAN_x4plus_anime_6B.pth` | BSD-3-Clause | illustrated / anime |

`lanczos` stays in the catalogue for the CI stand-in. It is not selectable.
The other catalogue entries stay for credits and the licence gate. They are
not selectable, so Run will not fetch them.

Both Nomos releases are Philip Hofmann's, on `github.com/Phhofm/models`. The
release text says `License: CC-BY-0.4`. That is not a Creative Commons
identifier. OpenModelDB's rights block for the same releases says CC-BY-4.0,
which is what the catalogue stores. The validator maps the typo `CC-BY-0.4` to
`CC-BY-4.0`. Those models need a line on the Credits screen.

`RealESRGAN_x4plus_anime_6B.pth` is the asset on Real-ESRGAN release `v0.2.2.4`.
The repository `LICENSE` is BSD-3-Clause, copyright 2021 Xintao Wang. The weight
file does not carry a separate licence. The catalogue uses BSD-3-Clause.

A release build rejects a non-commercial or missing licence before any byte is
fetched, including a file dropped in the custom models folder. GPL and other
licences outside the allow-list are rejected when `release=True`.

The neural models are native 4x. An x2 job is one 4x pass, then a resample
down to 2x. x4 is one pass. Double upscale at x4 is two passes (16x). The
output factor is 2, 4, or 16.

## Output files

The default container is AVIF at quality 60. The main set is AVIF, JPEG XL, and
WebP. JPEG XL is the lossless format (`lossless` on the request, or
`--lossless` on the CLI); the quality slider is ignored for that save. Lossy
JPEG XL maps the slider onto libjxl distance: 100 is 0, 80 is 1, 60 is 2.

JPEG stays available and is labelled legacy. HEIC is labelled legacy and is
listed only when a system codec is detected at runtime: a HEIF saver already
registered with Pillow, a system libheif that can encode HEVC, or `heif-enc`
on `PATH`. No HEIC encoder is a dependency. Windows Store HEVC is not probed.
PNG can be opened. Saving as PNG is refused.

Encoders that ship with the app:

| Format | Package | Licence |
|---|---|---|
| AVIF | Pillow >= 11.3 (libavif) | Pillow's own permissive build |
| JPEG XL | `imagecodecs` 2026.10.10 (libjxl) | BSD-3-Clause |
| WebP, JPEG | Pillow | permissive (libwebp, libjpeg-turbo) |

`pillow-jxl-plugin` is GPL-3 and is not used. `pillow-avif-plugin` and
`pillow-heif` are not dependencies. AVIF comes from Pillow itself.
`imagecodecs` 2026.10.10 publishes a cp312-abi3 wheel, which installs on
Python 3.14. The compare view decodes through Pillow, because Qt cannot
read AVIF or JPEG XL.

## Files on disk

Weights:

* Windows: `%LOCALAPPDATA%\Voxprint\models`
* Linux: `~/.local/share/voxprint/models` (or `$XDG_DATA_HOME/voxprint/models`)
* Override: `VOXPRINT_MODELS_DIR`, then `models_dir` in `suite.json`

`<models>/.users.json` is the shared refcount, the same file Audiobook Builder
and Movie Dubber write: `{"audiobook-builder": true, "movie-dubber": true}`.
Enlarge adds `"enlarge": true`. Unknown keys are kept. A damaged file is
treated as empty on the next write. `register` / `unregister` are the calls.
Unregister returns the other programs that are still true, so an uninstaller
can leave the folder alone while someone else uses it.

TensorRT engines are not refcounted and are safe to delete. They are rebuilt
from the ONNX file.

* Windows: `%LOCALAPPDATA%\Voxprint\trt-cache\<gpu>-<driver>-<trt>\`
* Linux: `~/.cache/voxprint/trt-cache/<gpu>-<driver>-<trt>\`
  (`$XDG_CACHE_HOME` is honoured)
* Override: `VOXPRINT_TRT_CACHE`

## Tiling

Tile size comes from `free_now - max(2 GB, 8% of total)`, measured again for
each batch. The ladder is 128, 256, 512, 768. A model hint is clamped to that.
A user tile from 32 to 2048 wins. Overlap is 16 px at 128 and above, 8 px from
64, and 0 below that.

Tiles overlap. The feather on an interior edge ramps across the overlap so the
two tiles add up to 1. A picture border does not fade. The right and bottom
edges are padded with edge pixels when the tile is short, and that padding is
cropped off the network output. An out-of-memory error or a system-memory spill
halves the tile (minimum 32) and retries. The next batch size is halved too,
and never goes below 1.

## Offline conversion

Runtime inference prefers a sibling `.onnx` next to the `.pth`. Build it with
spandrel's MIT package only:

```bash
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install spandrel onnx onnxruntime-gpu tensorrt-cu13
python tools/export_onnx.py path/to/model.pth
```

`tools/export_onnx.py` does not import `spandrel_extra_arches`. Copy the
`.onnx` into the shared models folder beside the weight (or pass `--out`).
The graph is fp16, with dynamic height and width. The author's own ONNX files
on the Nomos releases are fp32 and, for the fast model, a fixed 256 input.
This project does not use those files.

`requirements.txt` stays on the CPU ONNX Runtime wheel so CI can import the
package. The app still refuses to start on that machine. `requirements-gpu.txt`
is comments only. The NVIDIA driver must be from the 600 series or newer.
The wheels above are CUDA 13. The `cu126` index and the R560 driver branch are
not used.

## GPU benchmark

On a machine with a GPU provider and, for the PyTorch fallback, torch plus
spandrel:

```bash
python tools/benchmark_gpu.py --image samples/sample.png --model 4x-nomos2-realplksr-dysample --scale 2
```

The script downloads the pinned weight if it is missing, holds the suite GPU
lock for the run, and prints the provider, the size, the tile, and the
milliseconds. It needs the same NVIDIA GPU and 600-series driver the app
requires. A CPU ONNX timing run is `VOXPRINT_ORT_DEVICE=cpu` with a model
whose `.onnx` is already next to the weight (the unit test builds a tiny
Resize network instead of downloading).

CI runs that unit test. It does not run `benchmark_gpu.py`.
