# Shared inference engine

`voxprint_engine` is the image upscaler Voxprint AI Enlarge owns and Voxprint AI
Clarity imports. The installable distribution is `voxprint-engine` 1.1.0. It
does not import Qt. `API` is 1. The package version and `API` are different
numbers: the version marks this release, and `API` stays 1 until a breaking
change. 1.1.0 adds `ensure_onnx`, `iter_tile_frames`, and the optional
manifest fields `onnx_url`, `onnx_sha256`, and `execution`.

Clarity pins the merge commit of this release:

```text
voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.1.0
```

The previous pin stays valid for the 1.0 surface:

```text
voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0
```

The tag `engine-v1.0.0` is the earlier commit on `main`. `engine-v1.1.0`
publishes this file and the root `pyproject.toml`. The wheel contains
`voxprint_engine` only. The Qt app stays in this repository and is not part
of the wheel.

## Public API

`from voxprint_engine.api import ...` is the supported surface. The same
objects can be imported from the modules below. A pull request may add a
name. Removing one, or changing what it means, bumps `API`. The video
names below are that addition.

| Name | Role |
|---|---|
| `API`, `__version__` | Compatibility number and package version |
| `ModelManifest`, `Catalog`, `ManifestError` | One licence-bearing model, a catalogue, a rejected manifest |
| `validate_manifest`, `load_builtin`, `combined_catalog` | Check a manifest, read the built-in file, add a custom folder |
| `DEFAULT_MODEL_ID`, `LANCZOS_MODEL_ID` | The pinned fast photo model, and the CI stand-in id |
| `WorkerClient`, `WorkerError`, `UpscaleRequest`, `UpscaleResult` | JSON-line worker process |
| `RestoreBatchRequest` | Video worker op: NHWC frames in, restored frames out |
| `upscale_file`, `uses_lanczos` | One image, and whether this job must skip the network |
| `restore_frames`, `temporal_windows`, `tile_frames`, `iter_tile_frames` | Temporal windows, per-frame tiles, and a one-frame video iterator for Clarity |
| `ensure_weight`, `register`, `unregister` | Pinned download, and the shared `.users.json` refcount |
| `ensure_onnx` | Download a prebuilt ONNX file when the manifest pins one, otherwise convert the `.pth` to a checked fp16 ONNX file and cache it by sha256 |
| `models_dir`, `trt_cache_dir` | Weights folder and the rebuildable TensorRT cache |
| `vram_budget_gb`, `choose_tile`, `halve_tile`, `note_spill`, `sysmem_spill` | Suite VRAM margin, the tile ladder, and the 256 MB spill check |
| `read_shared_usage` | Windows shared GPU memory for one pid, or `None` |
| `gpu_job` | Hold `%TEMP%/voxprint-gpu.lock` for one job. `owner` defaults to `enlarge` |
| `decide` | Suite thermal pause rule |
| `shared_dir`, `read_manifest` | `%LOCALAPPDATA%\Voxprint\shared` or `~/.local/share/voxprint/shared`, and its refcounted manifest |
| `ensure_runtime`, `release_runtime` | Install or repair Python 3.14, the CUDA wheels, and the default model; drop one app and delete a payload only at zero holders |
| `ComponentStatus`, `RuntimeReport`, `SuiteInstallError` | One component, the check result (`as_dict` is the `--selftest --json` object), and a failed install |

```python
from voxprint_engine.api import (
    API,
    WorkerClient,
    UpscaleRequest,
    combined_catalog,
    gpu_job,
    upscale_file,
    validate_manifest,
)
```

Clarity should keep using the worker for a long job. `upscale_file` is the
same function the worker calls, for a benchmark or a one-off frame. `tests/test_api.py`
fails when a name in this table disappears. Clarity's video calls are on this
list and in the Video API below. `API` stays 1.

## What Run does

The worker is `python -m voxprint_engine.worker`. One JSON object per line.
`engine` on an upscale request is `auto` (the default) or `onnx`.

The window and `--upscale` start only when `nvidia-smi` reports an NVIDIA GPU
whose driver major version is 600 or newer. A computer without that GPU, or
with an older driver, does not open the window. `--upscale` prints the reason
and exits. `--capabilities --json` and `--serve --hidden` still start so a
client can read the refusal (`docs/IPC.md`). A card below compute capability
8.9 (RTX 40 / Ada) shows a diagnostics warning and still starts.

`VOXPRINT_CI_LANCZOS=1` skips that gate and resamples the image with Lanczos.
Tests and CI set the flag so they stay offline and GPU-free. It is not a
setting in the app. A user job never takes that path.

`VOXPRINT_ORT_DEVICE=cpu`, `device` `cpu-onnx`, or `engine` `onnx` runs the
CPU execution provider. That is the smoke test. It is not a user mode.

`--install-suite` pins `onnxruntime-gpu==1.31.0` and `tensorrt_cu13_libs==10.16.1.11`
(`--no-deps`). ORT 1.31 loads `nvinfer_10`. An unpinned `tensorrt-cu13` install
is TensorRT 11, and that provider does not load. The worker adds that library
directory with `os.add_dll_directory` on Windows and `LD_LIBRARY_PATH` plus
`RTLD_GLOBAL` on Linux before ONNX Runtime is imported. `--selftest` checks
that the TensorRT execution provider actually loads. The CI stand-in skips
that check.

Native `EP Error` text from ONNX Runtime is not protocol traffic. The worker
keeps JSON on a duplicated stdout and sends those prints to
`logs/worker-native.log`. The client also skips a line that does not start
with `{`. A TensorRT kernel that fails after the engine builds
(`InferenceError`) is retried on the CUDA execution provider.

Otherwise the session is:

1. TensorRT EP, fp16, engine cache enabled, unless the manifest says `execution: cuda` or `execution: tensorrt`
2. CUDA EP, when the manifest says `cuda`, when TensorRT is not installed, or when that TensorRT engine does not build

`4x-nomos2-hq-dat2` is `execution: cuda`. TensorRT builds an engine for that
network and then fails at runtime. The model step says so.

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
| `4x-nomos2-hq-dat2` | `4xNomos2_hq_dat2.pth` | CC-BY-4.0 | highest quality photos, CUDA provider |
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
| AVIF | Pillow >= 12.3.0 (libavif) | Pillow's own permissive build |
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

`<models>/.users.json` is the shared refcount for weight files, the same file
Audiobook Builder and Movie Dubber write: `{"audiobook-builder": true, "movie-dubber": true}`.
Enlarge adds `"enlarge": true`. Unknown keys are kept. A damaged file is
treated as empty on the next write. `register` / `unregister` are the calls.
Unregister returns the other programs that are still true, so an uninstaller
can leave the folder alone while someone else uses it.

The suite runtime is a second refcount, in `shared/manifest.json` under
`shared_dir()`. It tracks Python 3.14, the CUDA 13 wheels (torch, ONNX
Runtime GPU, TensorRT), and the default model. `ensure_runtime("enlarge")`
installs whatever is missing and records the app. `ensure_runtime("clarity")`
reuses the same files and adds a holder. `release_runtime` removes that one
holder and deletes a payload only when the list is empty. The default weight
stays in the models folder above; the manifest is what decides whether this
app may delete it. `VOXPRINT_CI_LANCZOS=1` skips the install and reports each
component as skipped. `API` stays 1.

TensorRT engines are not refcounted and are safe to delete. They are rebuilt
from the ONNX file.

* Windows: `%LOCALAPPDATA%\Voxprint\trt-cache\<gpu>-<driver>-<trt>\`
* Linux: `~/.cache/voxprint/trt-cache/<gpu>-<driver>-<trt>\`
  (`$XDG_CACHE_HOME` is honoured)
* Override: `VOXPRINT_TRT_CACHE`

The first TensorRT session reports `building the TensorRT engine`. A later
session that already finds a file in that folder reports `loading the cached
TensorRT engine`.

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

## Video API

Voxprint AI Clarity calls these helpers. Enlarge's image job leaves the new
arguments at their defaults, so `run_tile`, the `upscale` worker op, and the
compute-capability warning stay as they are. `API` stays 1. Scene cuts and
NVDEC are not part of this package. Clarity does not call `formats.py` or
`heic_system.py`.

```python
from voxprint_engine.hardware import assess_gpu
from voxprint_engine.gpu_lock import gpu_job
from voxprint_engine.session import open_session, run_batch
from voxprint_engine.vram import measure, choose_tile, note_spill, halve_tile, sysmem_spill
from voxprint_engine.video import temporal_windows, tile_frames, restore_frames
from voxprint_engine.worker.protocol import RestoreBatchRequest
```

### Startup and the GPU lock

`assess_gpu(index=0, runner=None, refuse_below=None)` is the Enlarge gate.
Omit `refuse_below` and a card below compute capability 8.9 still starts, with
the warning from the section above. Clarity passes `refuse_below=(8, 9)`. A
missing capability, or a card below that pair, then refuses to start. The
driver check still comes first: a driver below the 600 series refuses with the
same message as Enlarge. `VOXPRINT_CI_LANCZOS=1` still skips the gate. Clarity
does not honour that flag.

`try_acquire`, `release`, and `gpu_job` take `owner`. The default is
`enlarge`. Clarity passes `owner="clarity"`. One process has one owner.
A nested `gpu_job` in that process re-enters the lock the outer block already
holds and releases it only when the outer block ends. A worker whose parent
already holds the lock, with the same `owner`, shares that file: it does not
take a second lock and it does not delete the parent's. `eta` is written at
acquire as now plus `eta_s` (at least 60 seconds). The lock is stale when the
pid is gone, or when `eta` (else `started`) is more than two hours in the past.

### Session and VRAM

`open_session` is unchanged: TensorRT EP with fp16 and the engine cache under
the trt-cache directory, then CUDA EP when that engine does not build.
`session.run_batch(session, nchw)` runs one NCHW batch on that session and
returns float32 NCHW. When the session input type contains `float16`, the
batch is converted to fp16 before `session.run`. `run_tile` remains the image
path (one HWC tile in, one HWC tile out).

Clarity measures VRAM with the existing helpers, again before each batch:
`budget = max(0, free_now - max(2 GB, 8% of total))`, then `choose_tile`,
`note_spill`, and `halve_tile`. `sysmem_spill(shared_now, shared_baseline)` is
true when Windows shared GPU memory grew by more than 256 MB. `None` for
either sample means there is no spill, which is also what a missing counter
returns. `voxprint_engine.video.read_shared_usage` reads
`GPU Process Memory(pid_*)\Shared Usage` with `typeperf` on Windows and
returns `None` everywhere else. The thermal rule is the existing
`thermal.decide`: full speed for 2.5 hours, then pause between chunks while
the 5-minute median is at least 83 C or throttling exceeds 60 seconds, and
resume at or below 75 C.

### Windows, tiles, and the worker

`temporal_windows(frame_count, length, overlap)` slides a window of `length`
frames. `overlap` is left context and must be `>= 0` and `< length`. `length`
1 and `overlap` 0 is the frame-wise case: one window per frame, and every
frame is kept. A longer window pads with the edge frame outside the clip.
Across the list, the kept slots cover source frames `0 .. frame_count` once.
Padding slots are discarded.

`tile_frames` tiles each frame. The feather is the same ramp as `tile.py`:
interior overlaps sum to 1, and a picture border stays at full weight. A short
edge tile is padded by edge replication, and that padding is cropped off the
network output. The tiles of one step are stacked on N and passed to `infer`
as NCHW. `infer` returns NCHW at `scale` times the tile.

`restore_frames` gathers each window, runs `tile_frames`, and writes only the
kept rows back to the source index. Frame-wise models use `length=1` and
`window_overlap=0`.

The worker accepts `op` `restore_batch` on the same JSON-line stream as
`upscale`. `RestoreBatchRequest` (`id`, `frames_npy`, `out_npy`, `onnx`, and
optional `tile`, `overlap`, `batch`, `scale`, `length`, `window_overlap`,
`device`, `owner`) is defined in `voxprint_engine.video` and re-exported from
`voxprint_engine.worker.protocol`. The handler loads the NHWC float `.npy`
with `allow_pickle=False`, then holds `gpu_job` for `owner` (default
`clarity`) across the session build and `restore_frames`. The session is
`choices_for` (TensorRT when that provider is installed, fp16, trt-cache) and,
when that build fails, `cuda_fallback_choices`. The success line is `op`
`result`, `engine` `onnx`, with `dst` and `frames`. A missing file or a bad
array is `op` `error`. On the wire, a missing or zero `tile` is 512 and a
missing or zero `overlap` is 16. The image `upscale` op is unchanged and
still rejects a video file.

Weights stay in the shared store. Clarity's refcount key is `clarity`
(`register` / `unregister`). Downloads still finish before the GPU lock.

## ONNX conversion

The store downloads a `.pth`. The runtime needs a `.onnx`. `ensure_onnx`
produces it. The shared installer converts the default model after torch and
spandrel are installed. The Model step converts the others when they are
downloaded. A worker job converts too, when the weight is already on disk,
and it does that before taking the GPU lock. A headless job still refuses
with `model_missing` when the weight itself is absent. It does not download.

The cache file is `<models>/<sha256>.onnx`. A conversion is named with the
weight sha256. A prebuilt file (`onnx_url` and `onnx_sha256` on the manifest)
is preferred and named with its own sha256. A sibling `.onnx` next to the
weight is copied into that cache when `onnx.checker` accepts it.

Torch 2.14's default dynamo exporter writes two nodes with the same name.
`ensure_onnx` calls the legacy exporter (`dynamo=False`, opset 17, dynamic
height and width), converts the graph to fp16, and runs `onnx.checker`
before the cache file is kept. The loader is spandrel's MIT package.
`spandrel_extra_arches` is not imported.

```bash
python tools/export_onnx.py path/to/model.pth
```

That script calls `export_weight`. The install already has torch, spandrel,
and onnx. The author's own ONNX files on the Nomos releases are fp32 and, for
the fast model, a fixed 256 input. This project does not use those files.

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
